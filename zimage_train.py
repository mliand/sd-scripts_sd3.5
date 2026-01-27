import argparse
import math
import os
from multiprocessing import Value
from tqdm import tqdm

import torch
from accelerate.utils import set_seed
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import config_util, deepspeed_utils, strategy_base, train_util
from library.config_util import BlueprintGenerator, ConfigSanitizer
from library.custom_train_functions import apply_masked_loss, add_custom_train_arguments
from library.utils import setup_logging, add_logging_arguments
from library import strategy_zimage, zimage_train_utils, zimage_utils

setup_logging()
import logging

logger = logging.getLogger(__name__)


def get_noisy_model_input_and_timesteps(args, latents: torch.Tensor, noise: torch.Tensor, device, dtype):
    bsz = latents.shape[0]
    num_timesteps = 1000

    if args.timestep_sampling == "uniform":
        sigmas = torch.rand((bsz,), device=device)
    elif args.timestep_sampling == "sigmoid":
        sigmas = torch.sigmoid(args.sigmoid_scale * torch.randn((bsz,), device=device))
    else:
        sigmas = torch.randn((bsz,), device=device)
        sigmas = sigmas * args.sigmoid_scale
        sigmas = sigmas.sigmoid()
        shift = args.discrete_flow_shift
        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)

    timesteps = sigmas * num_timesteps
    sigmas = sigmas.view(-1, 1, 1, 1)
    noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
    return noisy_model_input.to(dtype), timesteps.to(dtype), sigmas


def train(args: argparse.Namespace):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    if args.pretrained_model_name_or_path is None:
        raise ValueError("--pretrained_model_name_or_path is required for Z-Image training.")
    if args.vae is None:
        raise ValueError("--vae is required for Z-Image training.")
    if args.text_encoder is None:
        raise ValueError("--text_encoder is required for Z-Image training.")
    cache_latents = args.cache_latents

    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        logger.warning("cache_text_encoder_outputs_to_disk enabled, enabling cache_text_encoder_outputs as well")
        args.cache_text_encoder_outputs = True
    if args.train_text_encoder and args.cache_text_encoder_outputs:
        raise ValueError("cache_text_encoder_outputs cannot be used when training the text encoder")

    if args.seed is not None:
        set_seed(args.seed)

    if args.cache_latents:
        latents_caching_strategy = strategy_zimage.ZImageLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0}".format(", ".join(ignored))
                )
        else:
            use_dreambooth_method = args.in_json is None
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)
        val_dataset_group = None

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(16)

    tokenizer_id = args.tokenizer or args.text_encoder
    zimage_tokenize_strategy = strategy_zimage.ZImageTokenizeStrategy(
        tokenizer_id,
        max_length=args.max_token_length,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
        apply_chat_template=not args.disable_chat_template,
    )
    strategy_base.TokenizeStrategy.set_strategy(zimage_tokenize_strategy)

    if args.debug_dataset:
        if args.cache_text_encoder_outputs:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(
                strategy_zimage.ZImageTextEncoderOutputsCachingStrategy(
                    args.cache_text_encoder_outputs_to_disk,
                    args.text_encoder_batch_size,
                    args.skip_cache_check,
                    False,
                    max_length=args.max_token_length,
                )
            )
        train_dataset_group.set_current_strategies()
        train_util.debug_dataset(train_dataset_group, True)
        return

    if len(train_dataset_group) == 0:
        logger.error("No data found. Please verify the metadata file and train_data_dir option.")
        return

    if cache_latents:
        assert train_dataset_group.is_latent_cacheable(), "when caching latents, color_aug or random_crop cannot be used"

    if args.cache_text_encoder_outputs:
        assert train_dataset_group.is_text_encoder_output_cacheable(), (
            "when caching text encoder output, caption_dropout_rate, shuffle_caption, token_warmup_step, "
            "caption_tag_dropout_rate cannot be used"
        )

    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # load VAE for caching latents
    vae = None
    if cache_latents:
        vae = zimage_utils.load_vae(args.vae, weight_dtype, "cpu")
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.eval()

        train_dataset_group.new_cache_latents(vae, accelerator)

        vae.to("cpu")
        clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    text_encoding_strategy = strategy_zimage.ZImageTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    text_encoder = zimage_utils.load_text_encoder(args.text_encoder, weight_dtype, "cpu")
    text_encoder.requires_grad_(args.train_text_encoder)
    text_encoder.eval()

    sample_prompts_te_outputs = None
    if args.cache_text_encoder_outputs:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder_caching_strategy = strategy_zimage.ZImageTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            is_partial=args.train_text_encoder,
            max_length=args.max_token_length,
        )
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_caching_strategy)

        with accelerator.autocast():
            train_dataset_group.new_cache_text_encoder_outputs([text_encoder], accelerator)

        if args.sample_prompts is not None:
            logger.info(f"cache Text Encoder outputs for sample prompt: {args.sample_prompts}")
            prompts = train_util.load_prompts(args.sample_prompts)
            sample_prompts_te_outputs = {}
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                        if p in sample_prompts_te_outputs:
                            continue
                        logger.info(f"cache Text Encoder outputs for prompt: {p}")
                        tokens_and_masks = zimage_tokenize_strategy.tokenize(p)
                        prompt_embeds, prompt_mask = text_encoding_strategy.encode_tokens(
                            zimage_tokenize_strategy, [text_encoder], tokens_and_masks
                        )
                        sample_prompts_te_outputs[p] = (prompt_embeds.cpu(), prompt_mask.cpu())

        text_encoder.to("cpu")
        clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    train_dataset_group.set_current_strategies()

    transformer = zimage_utils.load_transformer(args.pretrained_model_name_or_path, weight_dtype, "cpu")
    transformer.requires_grad_(True)
    transformer.train()

    params_to_optimize = [{"params": list(transformer.parameters()), "lr": args.learning_rate}]
    if args.train_text_encoder:
        params_to_optimize.append({"params": list(text_encoder.parameters()), "lr": args.learning_rate})

    _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)
    optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=min(args.max_data_loader_n_workers, os.cpu_count()),
        persistent_workers=args.persistent_data_loader_workers,
        pin_memory=True,
    )

    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(f"override steps for {args.max_train_epochs} epochs: {args.max_train_steps}")

    train_dataset_group.set_max_train_steps(args.max_train_steps)

    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    training_models = []
    if args.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(
            args,
            transformer=transformer,
            text_encoder=text_encoder if args.train_text_encoder else None,
        )
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [transformer]
        if args.train_text_encoder:
            text_encoder = accelerator.prepare(text_encoder)

    if not args.train_text_encoder and not args.cache_text_encoder_outputs:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if not cache_latents:
        vae = zimage_utils.load_vae(args.vae, weight_dtype, "cpu")
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.eval()

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info(f"Total train batch size (w. parallel & accumulation) = {total_batch_size}")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.max_train_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    global_step = 0
    epoch = 0

    optimizer_eval_fn()
    sample_text_encoder = None if sample_prompts_te_outputs is not None and args.cache_text_encoder_outputs else text_encoder
    zimage_train_utils.sample_images(
        accelerator, args, 0, global_step, transformer, vae, sample_text_encoder, sample_prompts_te_outputs
    )
    optimizer_train_fn()
    if len(accelerator.trackers) > 0:
        accelerator.log({}, step=0)

    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        transformer.train()
        if args.train_text_encoder:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            with accelerator.accumulate(*training_models):
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        latents = vae.encode(batch["images"].to(vae.dtype)).latent_dist.mode()
                        latents = latents.to(accelerator.device, dtype=weight_dtype)

                latents = zimage_utils.scale_shift_latents(latents, vae)

                text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                if text_encoder_outputs_list is not None:
                    prompt_embeds, prompt_mask = text_encoder_outputs_list
                    prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
                    prompt_mask = prompt_mask.to(accelerator.device).bool()
                else:
                    tokens_and_masks = batch["input_ids_list"]
                    input_ids = [ids.to(accelerator.device) for ids in tokens_and_masks]
                    prompt_embeds, prompt_mask = text_encoding_strategy.encode_tokens(
                        zimage_tokenize_strategy, [text_encoder], input_ids
                    )
                    prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
                    prompt_mask = prompt_mask.to(accelerator.device).bool()

                patch_size = transformer.all_patch_size[0] if hasattr(transformer, "all_patch_size") else 2
                image_sequence_length = (latents.shape[2] // patch_size) * (latents.shape[3] // patch_size)
                prompt_embeds, prompt_mask = zimage_train_utils._trim_pad_embeds_and_mask(
                    image_sequence_length, prompt_embeds, prompt_mask
                )

                cap_dtype = zimage_train_utils._get_model_param_dtype(transformer, transformer.dtype)
                cap_feats = [prompt_embeds[i][prompt_mask[i]].to(dtype=cap_dtype) for i in range(prompt_embeds.shape[0])]

                noise = torch.randn_like(latents)
                noisy_model_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(
                    args, latents, noise, accelerator.device, weight_dtype
                )

                t_input = (1000.0 - timesteps) / 1000.0
                t_input = t_input.to(accelerator.device, dtype=weight_dtype)

                noisy_model_input = noisy_model_input.unsqueeze(2)

                with accelerator.autocast():
                    zimage_train_utils._sync_pad_token_dtype(transformer)
                    model_pred = transformer(
                        x=noisy_model_input,
                        t=t_input,
                        cap_feats=cap_feats,
                    )

                model_pred = model_pred.squeeze(2)
                target = latents - noise

                huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, None)
                loss = train_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
                if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                    loss = apply_masked_loss(loss, batch)
                loss = loss.mean([1, 2, 3])

                loss_weights = batch["loss_weights"]
                loss = loss * loss_weights
                loss = loss.mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = []
                    for model in training_models:
                        params_to_clip.extend(model.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                optimizer_eval_fn()
                zimage_train_utils.sample_images(
                    accelerator, args, None, global_step, transformer, vae, sample_text_encoder, sample_prompts_te_outputs
                )

                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        zimage_train_utils.save_zimage_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(transformer),
                            accelerator.unwrap_model(text_encoder) if args.train_text_encoder else None,
                        )
                optimizer_train_fn()

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            zimage_train_utils.save_zimage_model_on_epoch_end_or_stepwise(
                args,
                True,
                accelerator,
                save_dtype,
                epoch,
                num_train_epochs,
                global_step,
                accelerator.unwrap_model(transformer),
                accelerator.unwrap_model(text_encoder) if args.train_text_encoder else None,
            )

        optimizer_eval_fn()
        zimage_train_utils.sample_images(
            accelerator, args, epoch + 1, global_step, transformer, vae, sample_text_encoder, sample_prompts_te_outputs
        )
        optimizer_train_fn()

        if global_step >= args.max_train_steps:
            break

    accelerator.end_training()
    optimizer_eval_fn()

    if accelerator.is_main_process:
        zimage_train_utils.save_zimage_model_on_train_end(
            args,
            save_dtype,
            epoch,
            global_step,
            accelerator.unwrap_model(transformer),
            accelerator.unwrap_model(text_encoder) if args.train_text_encoder else None,
        )
        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)
    train_util.add_dit_training_arguments(parser)
    zimage_train_utils.add_zimage_train_arguments(parser)

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train(args)
