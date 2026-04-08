# Minimum Inference Code for Z-Image

import argparse
import os
import random
import time

import torch
from PIL import Image

from library import strategy_zimage, train_util, zimage_utils
from library.device_utils import init_ipex, get_preferred_device
from library.utils import setup_logging
from library import zimage_train_utils

setup_logging()
import logging

logger = logging.getLogger(__name__)

init_ipex()


INFERENCE_BATCH_SIZE = 4


def _decoded_batch_to_pils(decoded: torch.Tensor) -> list[Image.Image]:
    image = (decoded / 2 + 0.5)
    image = torch.nan_to_num(image.detach().to(torch.float32), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1).cpu()

    if image.ndim == 3:
        image = image.unsqueeze(0)

    pil_images: list[Image.Image] = []
    for sample in image:
        sample_np = sample.permute(1, 2, 0).numpy()
        sample_np = (sample_np * 255).round().clip(0, 255).astype("uint8")
        pil_images.append(Image.fromarray(sample_np))
    return pil_images


def _make_2x2_grid(images: list[Image.Image]) -> Image.Image:
    if len(images) != INFERENCE_BATCH_SIZE:
        raise ValueError(f"Expected {INFERENCE_BATCH_SIZE} images, got {len(images)}")

    widths = [img.width for img in images]
    heights = [img.height for img in images]
    if len(set(widths)) != 1 or len(set(heights)) != 1:
        raise ValueError("All images must have the same size to create a 2x2 grid")

    cell_w = widths[0]
    cell_h = heights[0]
    grid = Image.new("RGB", (cell_w * 2, cell_h * 2))

    positions = [(0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h)]
    for img, pos in zip(images, positions):
        grid.paste(img, pos)

    return grid


def generate_image(
    transformer,
    vae,
    text_encoder,
    tokenize_strategy,
    encoding_strategy,
    prompt_dict,
    output_dir,
    output_name,
    steps,
    discrete_flow_shift,
    device,
    dtype,
):
    prompt: str = prompt_dict.get("prompt", "")
    negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", steps)
    width = prompt_dict.get("width", 512)
    height = prompt_dict.get("height", 512)
    guidance_scale = prompt_dict.get("guidance_scale", prompt_dict.get("scale", 4.0))
    seed = prompt_dict.get("seed")

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    seeds = [seed + i for i in range(INFERENCE_BATCH_SIZE)]
    logger.info(f"seeds: {seeds}")

    if negative_prompt is None:
        negative_prompt = ""

    height = max(64, height - height % 16)
    width = max(64, width - width % 16)
    logger.info(f"prompt: {prompt}")
    logger.info(f"negative_prompt: {negative_prompt}")
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"sample_steps: {sample_steps}")
    logger.info(f"guidance_scale: {guidance_scale}")

    prompt_embeds, prompt_mask = zimage_train_utils._encode_prompt(
        tokenize_strategy,
        encoding_strategy,
        text_encoder,
        prompt,
        None,
        device,
        dtype,
    )
    prompt_embeds = prompt_embeds.repeat(INFERENCE_BATCH_SIZE, 1, 1)
    prompt_mask = prompt_mask.repeat(INFERENCE_BATCH_SIZE, 1)

    do_cfg = guidance_scale is not None and guidance_scale > 1.0
    if do_cfg:
        negative_embeds, negative_mask = zimage_train_utils._encode_prompt(
            tokenize_strategy,
            encoding_strategy,
            text_encoder,
            negative_prompt,
            None,
            device,
            dtype,
        )
        negative_embeds = negative_embeds.repeat(INFERENCE_BATCH_SIZE, 1, 1)
        negative_mask = negative_mask.repeat(INFERENCE_BATCH_SIZE, 1)
    else:
        negative_embeds = None
        negative_mask = None

    channels = getattr(transformer, "in_channels", 16)
    latents = torch.stack(
        [
            torch.randn(
                (channels, height // 8, width // 8),
                device=device,
                dtype=torch.float32,
                generator=torch.Generator(device=device).manual_seed(sample_seed),
            )
            for sample_seed in seeds
        ],
        dim=0,
    )

    timesteps, sigmas = zimage_train_utils._get_timesteps_sigmas(sample_steps, discrete_flow_shift)
    timesteps = timesteps.to(device)
    sigmas = sigmas.to(device)

    with torch.autocast(device_type=device.type, dtype=dtype), torch.no_grad():
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000

            latent_model_input = latents.to(dtype).unsqueeze(2)
            model_out = transformer(x=latent_model_input, t=timestep, cap_feats=prompt_embeds, cap_mask=prompt_mask)

            if do_cfg:
                neg_out = transformer(x=latent_model_input, t=timestep, cap_feats=negative_embeds, cap_mask=negative_mask)
                noise_pred = model_out + guidance_scale * (model_out - neg_out)
            else:
                noise_pred = model_out

            noise_pred = -noise_pred.squeeze(2)
            latents = zimage_train_utils._step(noise_pred.to(torch.float32), latents, sigmas, i)

        latents = latents.to(vae.dtype)
        latents = zimage_train_utils._unscale_latents(latents, vae)
        decoded = zimage_train_utils._decode_latents(vae, latents)
        images = _decoded_batch_to_pils(decoded)
        image = _make_2x2_grid(images)

    os.makedirs(output_dir, exist_ok=True)
    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"{sample_steps:06d}"
    seed_suffix = f"_s{seeds[0]}-{seeds[-1]}"
    index = prompt_dict.get("enum", 0)
    filename = f"{'' if output_name is None else output_name + '_'}{num_suffix}_{index:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(output_dir, filename))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="path or HF id for Z-Image DiT")
    parser.add_argument("--vae", type=str, required=True, help="path to Z-Image VAE")
    parser.add_argument("--text_encoder", type=str, required=True, help="path or HF id for the Qwen text encoder")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer path or HF id (defaults to --text_encoder)")
    parser.add_argument("--max_token_length", type=int, default=512)
    parser.add_argument("--disable_chat_template", action="store_true")
    parser.add_argument("--prompt", type=str, default="A photo of a cat")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--discrete_flow_shift", type=float, default=3.0)
    parser.add_argument("--sample_prompts", type=str, default=None, help="prompt file (.txt/.toml/.json)")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--gate_type",
        type=str,
        default="none",
        choices=["headwise", "elementwise", "none"],
        help="Type of gating for attention: headwise, elementwise, or none. Default: none",
    )
    parser.add_argument(
        "--gate_layers",
        type=str,
        default=None,
        help="Layer indices (0-based) to enable gated attention in main layers. Accepts commas/spaces/ranges, e.g. '0,1 3-5'. Use 'all' for all layers",
    )
    parser.add_argument(
        "--gate_layers_noise_refiner",
        type=str,
        default=None,
        help="Deprecated and ignored. Gated attention is only applied to main transformer layers.",
    )
    parser.add_argument(
        "--gate_layers_context_refiner",
        type=str,
        default=None,
        help="Deprecated and ignored. Gated attention is only applied to main transformer layers.",
    )
    args = parser.parse_args()

    device = get_preferred_device()
    dtype = torch.float32
    if args.fp16:
        dtype = torch.float16
    elif args.bf16:
        dtype = torch.bfloat16

    logger.info("Loading Z-Image models...")
    transformer = zimage_utils.load_transformer(
        args.pretrained_model_name_or_path,
        dtype,
        device,
        gate_type=args.gate_type,
    )

    def _parse_layer_spec(text: str):
        if text in ("", "all"):
            return None
        parts = [p for p in text.replace(",", " ").split() if p]
        indices = []
        for part in parts:
            if "-" in part:
                start, end = part.split("-", 1)
                start = int(start)
                end = int(end)
                if end < start:
                    start, end = end, start
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(part))
        return sorted(set(indices))

    def _normalize_gate_layers(value):
        if value is None:
            return None
        if isinstance(value, str):
            return _parse_layer_spec(value)
        if isinstance(value, list):
            return sorted(set(int(x) for x in value))
        return value

    args.gate_layers = _normalize_gate_layers(args.gate_layers)
    args.gate_layers_noise_refiner = _normalize_gate_layers(args.gate_layers_noise_refiner)
    args.gate_layers_context_refiner = _normalize_gate_layers(args.gate_layers_context_refiner)

    if args.gate_layers_noise_refiner is not None or args.gate_layers_context_refiner is not None:
        logger.warning("Refiner gate layer options are ignored: gated attention is only applied to main transformer layers.")

    if any(
        v is not None
        for v in (args.gate_layers, args.gate_layers_noise_refiner, args.gate_layers_context_refiner)
    ) and hasattr(transformer, "set_gate_layers"):
        transformer.set_gate_layers(
            layer_ids=args.gate_layers,
        )
    vae = zimage_utils.load_vae(args.vae, dtype, device)
    text_encoder = zimage_utils.load_text_encoder(args.text_encoder, dtype, device)

    transformer.eval()
    vae.eval()
    text_encoder.eval()

    tokenize_strategy = strategy_zimage.ZImageTokenizeStrategy(
        args.tokenizer or args.text_encoder,
        max_length=args.max_token_length,
        tokenizer_cache_dir=None,
        apply_chat_template=not args.disable_chat_template,
    )
    encoding_strategy = strategy_zimage.ZImageTextEncodingStrategy()

    if args.sample_prompts is None:
        prompt_dict = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "sample_steps": args.steps,
            "width": args.width,
            "height": args.height,
            "enum": 0,
        }
        prompts = [prompt_dict]
    else:
        prompts = train_util.load_prompts(args.sample_prompts)

    save_dir = os.path.join(args.output_dir, "sample")
    for prompt_dict in prompts:
        generate_image(
            transformer,
            vae,
            text_encoder,
            tokenize_strategy,
            encoding_strategy,
            prompt_dict,
            save_dir,
            args.output_name,
            args.steps,
            args.discrete_flow_shift,
            device,
            dtype,
        )


if __name__ == "__main__":
    main()
