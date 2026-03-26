import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from library import deepspeed_utils, train_util
from library.magi_utils import (
    MagiModelWrapper,
    _str_to_torch_dtype,
    import_magi_components,
    load_jsonl_records,
    load_magi_configs,
    load_magi_dit_model,
    resolve_data_path,
)
from library.utils import setup_logging
from networks import lora_magi

setup_logging()
import logging

logger = logging.getLogger(__name__)
LATENT_CACHE_KEY_PATTERN = re.compile(r"^latents_\d+x\d+x\d+_[a-zA-Z0-9]+$")


@dataclass
class MagiCachedItem:
    latent_cache_path: str
    te_cache_path: str
    audio_cache_path: str | None = None


class MagiCachedDataset(Dataset):
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        records = load_jsonl_records(manifest_path)
        self.items: List[MagiCachedItem] = []
        audio_item_count = 0

        for rec in records:
            latent_cache = rec.get("magi_latent_cache")
            te_cache = rec.get("magi_te_cache")
            if not latent_cache or not te_cache:
                continue
            latent_path = resolve_data_path(manifest_path, str(latent_cache))
            te_path = resolve_data_path(manifest_path, str(te_cache))
            if not os.path.isfile(latent_path):
                raise FileNotFoundError(f"latent cache not found: {latent_path}")
            if not os.path.isfile(te_path):
                raise FileNotFoundError(f"text cache not found: {te_path}")
            audio_cache_path = None
            audio_cache = rec.get("magi_audio_cache")
            if audio_cache:
                resolved_audio_path = resolve_data_path(manifest_path, str(audio_cache))
                if not os.path.isfile(resolved_audio_path):
                    raise FileNotFoundError(f"audio cache not found: {resolved_audio_path}")
                audio_cache_path = resolved_audio_path
                audio_item_count += 1
            self.items.append(MagiCachedItem(latent_path, te_path, audio_cache_path))

        if len(self.items) == 0:
            raise ValueError("No valid items found in manifest. Ensure magi_latent_cache and magi_te_cache exist.")
        if 0 < audio_item_count < len(self.items):
            raise ValueError("Dataset mixes samples with and without magi_audio_cache. Please make the manifest consistent.")
        self.has_audio = audio_item_count == len(self.items)

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _pick_primary_latent_key(latent_sd: dict) -> str:
        if "latent_video" in latent_sd:
            return "latent_video"

        candidates = [k for k in latent_sd.keys() if LATENT_CACHE_KEY_PATTERN.match(str(k))]
        if not candidates:
            available = ", ".join(sorted(str(k) for k in latent_sd.keys()))
            raise KeyError(
                "No supported latent key found in cache. "
                f"Expected `latent_video` or `latents_FxHxW_dtype`, available keys: [{available}]"
            )

        candidates.sort()
        return candidates[0]

    def __getitem__(self, idx: int):
        item = self.items[idx]
        latent_sd = load_file(item.latent_cache_path)
        te_sd = load_file(item.te_cache_path)

        latent_key = self._pick_primary_latent_key(latent_sd)
        latent_video = latent_sd[latent_key]
        if latent_video.ndim == 5:
            latent_video = latent_video.squeeze(0)
        if latent_video.ndim == 3:
            latent_video = latent_video.unsqueeze(1)
        if latent_video.ndim != 4:
            raise ValueError(f"Unexpected latent shape {tuple(latent_video.shape)} in {item.latent_cache_path}, key={latent_key}")
        latent_video = latent_video.to(torch.float32).contiguous()

        prompt_embeds = te_sd["prompt_embeds"]
        if prompt_embeds.ndim == 3:
            prompt_embeds = prompt_embeds.squeeze(0)
        prompt_embeds = prompt_embeds.to(torch.float32).contiguous()

        if "prompt_len" in te_sd:
            prompt_len = int(te_sd["prompt_len"].reshape(-1)[0].item())
        else:
            prompt_len = int(prompt_embeds.shape[0])

        sample = {
            "latents": latent_video,
            "prompt_embeds": prompt_embeds,
            "prompt_len": prompt_len,
        }
        if item.audio_cache_path is not None:
            audio_sd = load_file(item.audio_cache_path)
            audio_latents = audio_sd["audio_latents"]
            if audio_latents.ndim == 3:
                audio_latents = audio_latents.squeeze(0)
            if audio_latents.ndim != 2:
                raise ValueError(f"Unexpected audio latent shape {tuple(audio_latents.shape)} in {item.audio_cache_path}")
            audio_len = int(audio_sd["audio_len"].reshape(-1)[0].item()) if "audio_len" in audio_sd else int(audio_latents.shape[0])
            sample["audio_latents"] = audio_latents.to(torch.float32).contiguous()
            sample["audio_len"] = audio_len
        return sample


def _collate(batch: Sequence[dict]):
    latent_shapes = [tuple(x["latents"].shape) for x in batch]
    if len(set(latent_shapes)) != 1:
        raise ValueError(f"All latent shapes in a batch must match. Got: {latent_shapes}")

    te_shapes = [tuple(x["prompt_embeds"].shape) for x in batch]
    if len(set(te_shapes)) != 1:
        raise ValueError(f"All prompt embed shapes in a batch must match. Got: {te_shapes}")

    latents = torch.stack([x["latents"] for x in batch], dim=0)
    prompt_embeds = torch.stack([x["prompt_embeds"] for x in batch], dim=0)
    prompt_len = torch.tensor([int(x["prompt_len"]) for x in batch], dtype=torch.int32)

    collated = {
        "latents": latents,
        "prompt_embeds": prompt_embeds,
        "prompt_len": prompt_len,
    }
    has_audio = any("audio_latents" in x for x in batch)
    if has_audio:
        if not all("audio_latents" in x for x in batch):
            raise ValueError("Mixed audio/non-audio samples in the same dataset are not supported.")
        audio_lengths = [int(x["audio_len"]) for x in batch]
        max_audio_len = max(audio_lengths)
        audio_channels = int(batch[0]["audio_latents"].shape[-1])
        audio_latents = torch.zeros((len(batch), max_audio_len, audio_channels), dtype=torch.float32)
        for i, x in enumerate(batch):
            cur = x["audio_latents"]
            audio_latents[i, : cur.shape[0], :] = cur
        collated["audio_latents"] = audio_latents
        collated["audio_len"] = torch.tensor(audio_lengths, dtype=torch.int32)
    return collated


def _parse_train_audio_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in {"auto", "always", "never"}:
        raise ValueError(f"Unsupported --train_audio mode: {mode}")
    return value


def _resolve_audio_training(mode: str, dataset_has_audio: bool) -> bool:
    if mode == "always":
        if not dataset_has_audio:
            raise ValueError("--train_audio=always but dataset manifest has no magi_audio_cache entries.")
        return True
    if mode == "never":
        return False
    return dataset_has_audio


def _sample_sigmas(batch_size: int, device: torch.device, shift: float) -> torch.Tensor:
    sigmas = torch.rand((batch_size, 1, 1, 1, 1), device=device)
    if shift != 1.0:
        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
    return sigmas


def _save_model_checkpoint(
    accelerator: Accelerator,
    wrapper: MagiModelWrapper,
    output_dir: str,
    output_name: str,
    step: int,
    save_dtype: torch.dtype,
):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"{output_name}-step{step:08d}.safetensors")

    unwrapped = accelerator.unwrap_model(wrapper)
    model = unwrapped.model

    state_dict = {}
    for key, value in model.state_dict().items():
        if value.dtype != save_dtype:
            value = value.detach().to("cpu").to(save_dtype)
        else:
            value = value.detach().to("cpu")
        state_dict[key] = value.contiguous()

    save_file(state_dict, ckpt_path)
    logger.info(f"Saved checkpoint: {ckpt_path}")


def _save_lora_checkpoint(
    accelerator: Accelerator,
    wrapper: MagiModelWrapper,
    output_dir: str,
    output_name: str,
    step: int,
    save_dtype: torch.dtype,
):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"{output_name}-lora-step{step:08d}.safetensors")

    unwrapped = accelerator.unwrap_model(wrapper)
    lora_sd = lora_magi.get_lora_state_dict(unwrapped.model)
    if len(lora_sd) == 0:
        raise RuntimeError("No LoRA weights found in model. Did you enable --train_lora?")

    cast_sd = {}
    for key, value in lora_sd.items():
        if value.is_floating_point():
            cast_sd[key] = value.to(save_dtype).contiguous()
        else:
            cast_sd[key] = value.contiguous()

    save_file(cast_sd, ckpt_path)
    logger.info(f"Saved LoRA checkpoint: {ckpt_path}")


def train(args: argparse.Namespace) -> None:
    if not hasattr(args, "max_data_loader_n_workers"):
        args.max_data_loader_n_workers = args.dataloader_num_workers
    deepspeed_utils.prepare_deepspeed_args(args)
    if args.deepspeed and hasattr(args, "max_data_loader_n_workers"):
        args.dataloader_num_workers = min(args.dataloader_num_workers, int(args.max_data_loader_n_workers))

    if args.seed is not None:
        set_seed(args.seed)

    deepspeed_plugin = deepspeed_utils.prepare_deepspeed_plugin(args)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        deepspeed_plugin=deepspeed_plugin,
    )

    model_dtype = _str_to_torch_dtype(args.model_dtype, default=torch.bfloat16)
    save_dtype = _str_to_torch_dtype(args.save_dtype, default=torch.bfloat16)

    model_cfg, data_proxy_cfg, _ = load_magi_configs(args.config_load_path)
    comps = import_magi_components()
    MagiDataProxy = comps["MagiDataProxy"]

    if data_proxy_cfg.frame_receptive_field != -1 and args.train_batch_size > 1:
        raise ValueError(
            "MagiDataProxy local attention requires batch_size=1 when frame_receptive_field != -1. "
            "Set --train_batch_size 1, or use a config with frame_receptive_field=-1."
        )

    dataset = MagiCachedDataset(args.dataset_jsonl)
    use_audio_training = _resolve_audio_training(_parse_train_audio_mode(args.train_audio), dataset.has_audio)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate,
    )

    model = load_magi_dit_model(
        args.pretrained_model_name_or_path,
        model_cfg,
        accelerator.device,
        model_dtype,
    )
    model.train()

    if args.train_lora:
        model.requires_grad_(False)
        lora_alpha = float(args.lora_alpha) if args.lora_alpha is not None else float(args.lora_rank)
        lora_network = lora_magi.create_network(
            multiplier=1.0,
            network_dim=args.lora_rank,
            network_alpha=lora_alpha,
            vae=None,
            text_encoder=None,
            unet=model,
            neuron_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            train_adapter=args.lora_train_adapter,
        )
        lora_network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
        replaced = len(lora_network.unet_loras)
        if replaced <= 0:
            raise ValueError(
                f"No modules matched LoRA targets: {args.lora_target_modules}. "
                "Please check --lora_target_modules against DiT module names."
            )
        trainable_params = list(lora_network.parameters())
        logger.info(
            "Enabled LoRA training: "
            f"modules={replaced}, rank={args.lora_rank}, alpha={lora_alpha}, dropout={args.lora_dropout}, "
            f"trainable_params={sum(p.numel() for p in trainable_params)}"
        )
    else:
        for p in model.parameters():
            p.requires_grad = True
        trainable_params = list(model.parameters())

    data_proxy = MagiDataProxy(data_proxy_cfg)
    wrapper = MagiModelWrapper(model, data_proxy, audio_in_channels=model_cfg.audio_in_channels)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.max_train_steps is None:
        if args.max_train_epochs is None:
            raise ValueError("Either --max_train_steps or --max_train_epochs must be specified.")
        updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.max_train_epochs * updates_per_epoch

    warmup_steps = max(0, int(args.lr_warmup_steps))

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(args, magi=wrapper)
        ds_model, optimizer, dataloader, scheduler = accelerator.prepare(ds_model, optimizer, dataloader, scheduler)
        training_models = [ds_model]
    else:
        wrapper, optimizer, dataloader, scheduler = accelerator.prepare(wrapper, optimizer, dataloader, scheduler)
        training_models = [wrapper]

    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    logger.info(
        "Start Magi training: "
        f"items={len(dataset)}, batches/epoch={len(dataloader)}, max_steps={args.max_train_steps}, "
        f"batch_size={args.train_batch_size}, grad_accum={args.gradient_accumulation_steps}, deepspeed={args.deepspeed}, "
        f"use_audio_training={use_audio_training}"
    )

    progress = tqdm(total=args.max_train_steps, desc="magi_train", disable=not accelerator.is_local_main_process)
    global_step = 0
    running_loss = 0.0

    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(*training_models):
                latents = batch["latents"].to(device=accelerator.device, dtype=model_dtype)
                prompt_embeds = batch["prompt_embeds"].to(device=accelerator.device, dtype=model_dtype)
                prompt_len = batch["prompt_len"].to(device=accelerator.device)

                noise = torch.randn_like(latents)
                sigmas = _sample_sigmas(latents.shape[0], latents.device, args.discrete_flow_shift).to(dtype=latents.dtype)
                noisy = (1.0 - sigmas) * latents + sigmas * noise
                noisy_audio = None
                audio_lengths = None
                audio_loss = None
                if use_audio_training:
                    audio_latents = batch["audio_latents"].to(device=accelerator.device, dtype=model_dtype)
                    audio_lengths = batch["audio_len"].to(device=accelerator.device)
                    audio_noise = torch.randn_like(audio_latents)
                    audio_sigmas = sigmas.squeeze(-1).squeeze(-1)
                    noisy_audio = (1.0 - audio_sigmas) * audio_latents + audio_sigmas * audio_noise

                pred_video, pred_audio = wrapper(
                    noisy_video=noisy,
                    text_embeds=prompt_embeds,
                    text_lengths=[int(x) for x in prompt_len.tolist()],
                    noisy_audio=noisy_audio,
                    audio_lengths=None if audio_lengths is None else [int(x) for x in audio_lengths.tolist()],
                )

                target_video = noise - latents
                video_loss = F.mse_loss(pred_video.float(), target_video.float(), reduction="mean")
                loss = video_loss
                if use_audio_training:
                    target_audio = audio_noise - audio_latents
                    audio_mask = (
                        torch.arange(audio_latents.shape[1], device=audio_latents.device).unsqueeze(0)
                        < audio_lengths.unsqueeze(1)
                    ).unsqueeze(-1).to(pred_audio.dtype)
                    pred_audio_masked = pred_audio * audio_mask
                    target_audio_masked = target_audio * audio_mask
                    denom = (audio_mask.sum() * pred_audio.shape[-1]).clamp_min(1.0)
                    audio_loss = ((pred_audio_masked.float() - target_audio_masked.float()) ** 2).sum() / denom
                    loss = loss + float(args.audio_loss_weight) * audio_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    params_to_clip = []
                    for m in training_models:
                        params_to_clip.extend(m.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                running_loss += float(loss.detach().item())
                progress.update(1)

                if global_step % args.log_every_n_steps == 0 and accelerator.is_local_main_process:
                    avg_loss = running_loss / args.log_every_n_steps
                    running_loss = 0.0
                    lr = scheduler.get_last_lr()[0]
                    if audio_loss is None:
                        logger.info(f"step={global_step} loss={avg_loss:.6f} video_loss={video_loss.detach().item():.6f} lr={lr:.6e}")
                    else:
                        logger.info(
                            f"step={global_step} loss={avg_loss:.6f} video_loss={video_loss.detach().item():.6f} "
                            f"audio_loss={audio_loss.detach().item():.6f} lr={lr:.6e}"
                        )

                if args.save_every_n_steps > 0 and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_local_main_process:
                        if args.train_lora:
                            _save_lora_checkpoint(
                                accelerator=accelerator,
                                wrapper=wrapper,
                                output_dir=args.output_dir,
                                output_name=args.output_name,
                                step=global_step,
                                save_dtype=save_dtype,
                            )
                        else:
                            _save_model_checkpoint(
                                accelerator=accelerator,
                                wrapper=wrapper,
                                output_dir=args.output_dir,
                                output_name=args.output_name,
                                step=global_step,
                                save_dtype=save_dtype,
                            )
                    accelerator.wait_for_everyone()

                if global_step >= args.max_train_steps:
                    break

        if len(dataloader) == 0:
            raise RuntimeError("Empty dataloader.")

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        if args.train_lora:
            _save_lora_checkpoint(
                accelerator=accelerator,
                wrapper=wrapper,
                output_dir=args.output_dir,
                output_name=args.output_name,
                step=global_step,
                save_dtype=save_dtype,
            )
        else:
            _save_model_checkpoint(
                accelerator=accelerator,
                wrapper=wrapper,
                output_dir=args.output_dir,
                output_name=args.output_name,
                step=global_step,
                save_dtype=save_dtype,
            )

    logger.info("Training completed.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Native daVinci-MagiHuman training support for sd-scripts (cached latent/TE workflow).")

    parser.add_argument("--dataset_jsonl", type=str, required=True, help="Manifest JSONL containing magi_latent_cache and magi_te_cache fields.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Path to daVinci DiT checkpoint (dir or .safetensors).")
    parser.add_argument("--config_load_path", type=str, default=None, help="Optional daVinci config.json for arch/data_proxy settings.")

    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints.")
    parser.add_argument("--output_name", type=str, default="magi", help="Checkpoint filename prefix.")

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_warmup_steps", type=int, default=0)

    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_train_epochs", type=int, default=None)

    parser.add_argument("--save_every_n_steps", type=int, default=500)
    parser.add_argument("--log_every_n_steps", type=int, default=10)

    parser.add_argument("--model_dtype", type=str, default="bf16", help="bf16/fp16/fp32")
    parser.add_argument("--save_dtype", type=str, default="bf16", help="bf16/fp16/fp32")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--full_fp16", action="store_true", help="Enable full fp16 training patch (same as other sd-scripts trainers).")

    parser.add_argument("--discrete_flow_shift", type=float, default=5.0, help="Shift used in flow-style sigma sampling.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_audio", type=str, default="auto", help="Audio training mode: auto/always/never.")
    parser.add_argument("--audio_loss_weight", type=float, default=1.0, help="Weight applied to audio MSE when audio training is enabled.")

    parser.add_argument("--train_lora", action="store_true", help="Enable LoRA training (freeze base DiT, train LoRA only).")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="linear_qkv,linear_proj,up_gate_proj,down_proj",
        help="Comma-separated module-name substrings to apply LoRA.",
    )
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=float, default=None, help="LoRA alpha (default: same as rank).")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout.")
    parser.add_argument("--lora_train_adapter", action="store_true", help="Also apply LoRA to DiT adapter modules.")

    deepspeed_utils.add_deepspeed_arguments(parser)

    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
