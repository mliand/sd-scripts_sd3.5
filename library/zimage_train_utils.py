import argparse
import os
import time
from typing import Optional

import numpy as np
from PIL import Image

import torch
from safetensors.torch import save_file

from accelerate import Accelerator, PartialState

from library import strategy_base, train_util
from library.device_utils import clean_memory_on_device

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def save_models(
    ckpt_path: str,
    transformer: torch.nn.Module,
    text_encoder: Optional[torch.nn.Module],
    save_dtype: Optional[torch.dtype] = None,
):
    state_dict = {}
    for key, value in transformer.state_dict().items():
        if save_dtype is not None and value.dtype != save_dtype:
            value = value.detach().clone().to("cpu").to(save_dtype)
        state_dict[key] = value

    save_file(state_dict, ckpt_path)

    if text_encoder is not None:
        text_encoder_path = ckpt_path.replace(".safetensors", "_text_encoder.safetensors")
        te_state_dict = text_encoder.state_dict()
        save_file(te_state_dict, text_encoder_path)


def save_zimage_model_on_train_end(
    args: argparse.Namespace,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    transformer: torch.nn.Module,
    text_encoder: Optional[torch.nn.Module],
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        save_models(ckpt_file, transformer, text_encoder, save_dtype)

    from library import train_util

    train_util.save_sd_model_on_train_end_common(args, True, True, epoch, global_step, sd_saver, None)


def save_zimage_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    transformer: torch.nn.Module,
    text_encoder: Optional[torch.nn.Module],
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        save_models(ckpt_file, transformer, text_encoder, save_dtype)

    from library import train_util

    train_util.save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        True,
        True,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        None,
    )


def add_zimage_train_arguments(parser: argparse.ArgumentParser):
    if "--vae" not in parser._option_string_actions:
        parser.add_argument(
            "--vae",
            type=str,
            required=True,
            help="path to Z-Image VAE (diffusers directory or safetensors file)",
        )
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="path or HF id for the Qwen text encoder",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="tokenizer path or HF id (defaults to --text_encoder)",
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=512,
        help="maximum token length for Qwen tokenizer",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="enable training of the Qwen text encoder",
    )
    parser.add_argument(
        "--disable_chat_template",
        action="store_true",
        help="do not apply tokenizer chat template for prompts",
    )
    parser.add_argument(
        "--timestep_sampling",
        type=str,
        default="shift",
        choices=["uniform", "sigmoid", "shift"],
        help="timestep sampling method for Z-Image training",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=3.0,
        help="discrete flow shift for shift-based timestep sampling",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help="sigmoid scale for timestep sampling (used by sigmoid/shift)",
    )


def _get_timesteps_sigmas(num_inference_steps: int, shift: float) -> tuple[torch.Tensor, torch.Tensor]:
    num_train_timesteps = 1000
    timesteps = np.linspace(num_train_timesteps, 1, num_inference_steps + 1)[:-1]
    sigmas = timesteps / num_train_timesteps

    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = sigmas * num_train_timesteps

    timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
    sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
    sigmas = torch.cat([sigmas, torch.zeros(1, dtype=sigmas.dtype, device=sigmas.device)], dim=0)

    return timesteps, sigmas


def _step(model_output: torch.Tensor, sample: torch.Tensor, sigmas: torch.Tensor, step_index: int) -> torch.Tensor:
    sample = sample.to(torch.float32)
    sigma = sigmas[step_index]
    sigma_next = sigmas[step_index + 1]
    dt = sigma_next - sigma
    prev_sample = sample + dt * model_output
    return prev_sample.to(model_output.dtype)


def _unscale_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    scale = getattr(vae.config, "scaling_factor", 1.0)
    shift = getattr(vae.config, "shift_factor", 0.0)
    return (latents / scale) + shift


def _decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latents)
    if hasattr(decoded, "sample"):
        decoded = decoded.sample
    return decoded


def _latents_to_pil(latents: torch.Tensor) -> Image.Image:
    image = (latents / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu()
    if image.ndim == 4:
        image = image[0]
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).round().clip(0, 255).astype("uint8")
    return Image.fromarray(image)


def _encode_prompt(tokenize_strategy, encoding_strategy, text_encoder, prompt: str, cached_outputs, device, dtype):
    if cached_outputs is not None and prompt in cached_outputs:
        prompt_embeds, prompt_mask = cached_outputs[prompt]
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        prompt_mask = prompt_mask.to(device=device).bool()
    else:
        tokens_and_masks = tokenize_strategy.tokenize(prompt)
        prompt_embeds, prompt_mask = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens_and_masks)
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        prompt_mask = prompt_mask.to(device=device).bool()

    if prompt_embeds.shape[0] == 1:
        actual_length = int(prompt_mask.sum(dim=1).item())
        prompt_embeds = prompt_embeds[:, :actual_length, :]
        prompt_mask = prompt_mask[:, :actual_length]

    return prompt_embeds, prompt_mask


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    transformer,
    vae,
    text_encoder,
    sample_prompts_te_outputs,
    prompt_replacement=None,
):
    if steps == 0:
        if not args.sample_at_first:
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:
                return

    logger.info("")
    logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
    if args.sample_prompts is None:
        logger.error("No prompt file / プロンプトファイルがありません")
        return
    if not os.path.isfile(args.sample_prompts) and sample_prompts_te_outputs is None:
        logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    distributed_state = PartialState()

    transformer = accelerator.unwrap_model(transformer)
    text_encoder = None if text_encoder is None else accelerator.unwrap_model(text_encoder)

    org_vae_device = vae.device
    vae.to(distributed_state.device)

    prompts = train_util.load_prompts(args.sample_prompts)
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    if distributed_state.num_processes <= 1:
        with torch.no_grad(), accelerator.autocast():
            for prompt_dict in prompts:
                sample_image_inference(
                    accelerator,
                    args,
                    transformer,
                    vae,
                    text_encoder,
                    save_dir,
                    prompt_dict,
                    epoch,
                    steps,
                    sample_prompts_te_outputs,
                    prompt_replacement,
                )
    else:
        per_process_prompts = []
        for i in range(distributed_state.num_processes):
            per_process_prompts.append(prompts[i :: distributed_state.num_processes])

        with torch.no_grad():
            with distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
                for prompt_dict in prompt_dict_lists[0]:
                    sample_image_inference(
                        accelerator,
                        args,
                        transformer,
                        vae,
                        text_encoder,
                        save_dir,
                        prompt_dict,
                        epoch,
                        steps,
                        sample_prompts_te_outputs,
                        prompt_replacement,
                    )

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    vae.to(org_vae_device)
    clean_memory_on_device(accelerator.device)


def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    transformer,
    vae,
    text_encoder,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
):
    assert isinstance(prompt_dict, dict)
    prompt: str = prompt_dict.get("prompt", "")
    negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", 30)
    width = prompt_dict.get("width", 512)
    height = prompt_dict.get("height", 512)
    guidance_scale = prompt_dict.get("guidance_scale", prompt_dict.get("scale", 7.5))
    seed = prompt_dict.get("seed")

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt is not None:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    else:
        torch.seed()
        if torch.cuda.is_available():
            torch.cuda.seed()

    if negative_prompt is None:
        negative_prompt = ""

    height = max(64, height - height % 8)
    width = max(64, width - width % 8)
    logger.info(f"prompt: {prompt}")
    logger.info(f"negative_prompt: {negative_prompt}")
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"sample_steps: {sample_steps}")
    logger.info(f"guidance_scale: {guidance_scale}")
    if seed is not None:
        logger.info(f"seed: {seed}")

    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

    device = accelerator.device
    dtype = transformer.dtype

    prompt_embeds, prompt_mask = _encode_prompt(
        tokenize_strategy,
        encoding_strategy,
        text_encoder,
        prompt,
        sample_prompts_te_outputs,
        device,
        dtype,
    )

    do_cfg = guidance_scale is not None and guidance_scale > 1.0
    if do_cfg:
        negative_embeds, negative_mask = _encode_prompt(
            tokenize_strategy,
            encoding_strategy,
            text_encoder,
            negative_prompt,
            sample_prompts_te_outputs,
            device,
            dtype,
        )
    else:
        negative_embeds = None
        negative_mask = None

    channels = getattr(transformer, "in_channels", 16)
    latents = torch.randn(
        (1, channels, height // 8, width // 8),
        device=device,
        dtype=torch.float32,
        generator=torch.Generator(device=device).manual_seed(seed) if seed is not None else None,
    )

    timesteps, sigmas = _get_timesteps_sigmas(sample_steps, args.discrete_flow_shift)
    timesteps = timesteps.to(device)
    sigmas = sigmas.to(device)

    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0])
        timestep = (1000 - timestep) / 1000

        latent_model_input = latents.to(dtype).unsqueeze(2)
        model_out = transformer(x=latent_model_input, t=timestep, cap_feats=prompt_embeds, cap_mask=prompt_mask)

        if do_cfg:
            neg_out = transformer(x=latent_model_input, t=timestep, cap_feats=negative_embeds, cap_mask=negative_mask)
            noise_pred = neg_out + guidance_scale * (model_out - neg_out)
        else:
            noise_pred = model_out

        noise_pred = -noise_pred.squeeze(2)
        latents = _step(noise_pred.to(torch.float32), latents, sigmas, i)

    latents = latents.to(vae.dtype)
    latents = _unscale_latents(latents, vae)
    decoded = _decode_latents(vae, latents)
    image = _latents_to_pil(decoded)

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = prompt_dict["enum"]
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))
