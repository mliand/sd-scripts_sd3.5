# SD3.5 inference helper for sd-scripts (MMDiT only)

import argparse
import datetime
import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.amp
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModelWithProjection, T5EncoderModel

from library.device_utils import init_ipex, get_preferred_device

init_ipex()

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import sd3_models, sd3_utils, strategy_sd3
from library.utils import load_safetensors


def get_noise(seed: int, latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device)
    generator.manual_seed(seed)
    return torch.randn(latent.size(), dtype=latent.dtype, layout=latent.layout, generator=generator, device=device)


def get_sigmas(sampling: sd3_utils.ModelSamplingDiscreteFlow, steps: int) -> torch.Tensor:
    start = sampling.timestep(sampling.sigma_max)
    end = sampling.timestep(sampling.sigma_min)
    timesteps = torch.linspace(start, end, steps)
    sigs = []
    for ts in timesteps:
        sigs.append(sampling.sigma(ts))
    sigs.append(0.0)
    return torch.tensor(sigs, dtype=torch.float32)


def max_denoise(model_sampling: sd3_utils.ModelSamplingDiscreteFlow, sigmas: torch.Tensor) -> bool:
    max_sigma = float(model_sampling.sigma_max)
    sigma = float(sigmas[0])
    return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma


def do_sample(
    *,
    vae: sd3_models.SDVAE,
    mmdit: sd3_models.MMDiT,
    height: int,
    width: int,
    initial_latent: Optional[torch.Tensor],
    seed: int,
    cond: Tuple[torch.Tensor, torch.Tensor],
    neg_cond: Tuple[torch.Tensor, torch.Tensor],
    steps: int,
    cfg_scale: float,
    dtype: torch.dtype,
    device: torch.device,
):
    if initial_latent is None:
        latent = torch.zeros(1, 16, height // 8, width // 8, device=device)
    else:
        latent = initial_latent

    latent = latent.to(dtype).to(device)
    noise = get_noise(seed, latent, device)

    model_sampling = sd3_utils.ModelSamplingDiscreteFlow(shift=3.0)  # 3.0 for SD3/SD3.5
    sigmas = get_sigmas(model_sampling, steps).to(device)

    noise_scaled = model_sampling.noise_scaling(sigmas[0], noise, latent, max_denoise(model_sampling, sigmas))

    c_crossattn = torch.cat([cond[0], neg_cond[0]]).to(device).to(dtype)
    y = torch.cat([cond[1], neg_cond[1]]).to(device).to(dtype)

    x = noise_scaled.to(device).to(dtype)

    with torch.no_grad():
        for i in tqdm(range(len(sigmas) - 1)):
            sigma_hat = sigmas[i]

            timestep = model_sampling.timestep(sigma_hat).float()
            timestep = torch.tensor([timestep, timestep], device=device)

            x_c_nc = torch.cat([x, x], dim=0)

            with torch.autocast(device_type=device.type, dtype=dtype):
                model_output = mmdit(x_c_nc, timestep, context=c_crossattn, y=y)
            model_output = model_output.float()

            batched = model_sampling.calculate_denoised(sigma_hat, model_output, x)
            pos_out, neg_out = batched.chunk(2)
            denoised = neg_out + (pos_out - neg_out) * cfg_scale

            dims_to_append = x.ndim - sigma_hat.ndim
            sigma_hat_dims = sigma_hat[(...,) + (None,) * dims_to_append]
            d = (x - denoised) / sigma_hat_dims

            dt = sigmas[i + 1] - sigma_hat
            x = (x + d * dt).to(dtype)

    latent = x
    latent = vae.process_out(latent)
    return latent


def generate_image(
    *,
    mmdit: sd3_models.MMDiT,
    vae: sd3_models.SDVAE,
    clip_l: CLIPTextModelWithProjection,
    clip_g: CLIPTextModelWithProjection,
    t5xxl: T5EncoderModel,
    tokenize_strategy: strategy_sd3.Sd3TokenizeStrategy,
    encoding_strategy: strategy_sd3.Sd3TextEncodingStrategy,
    steps: int,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    cfg_scale: float,
    dtype: torch.dtype,
    device: torch.device,
    offload: bool,
    output_dir: str,
):
    logger.info("Encoding prompts...")

    clip_l.to(device)
    clip_g.to(device)
    t5xxl.to(device)

    with torch.autocast(device_type=device.type, dtype=dtype), torch.no_grad():
        tokens_and_masks = tokenize_strategy.tokenize(prompt)
        lg_out, t5_out, pooled, *_ = encoding_strategy.encode_tokens(
            tokenize_strategy,
            [clip_l, clip_g, t5xxl],
            tokens_and_masks,
            apply_lg_attn_mask=False,
            apply_t5_attn_mask=False,
            enable_dropout=False,
        )
        cond = encoding_strategy.concat_encodings(lg_out, t5_out, pooled)

        tokens_and_masks = tokenize_strategy.tokenize(negative_prompt)
        lg_out, t5_out, pooled, *_ = encoding_strategy.encode_tokens(
            tokenize_strategy,
            [clip_l, clip_g, t5xxl],
            tokens_and_masks,
            apply_lg_attn_mask=False,
            apply_t5_attn_mask=False,
            enable_dropout=False,
        )
        neg_cond = encoding_strategy.concat_encodings(lg_out, t5_out, pooled)

    if offload:
        clip_l.to("cpu")
        clip_g.to("cpu")
        t5xxl.to("cpu")

    logger.info("Sampling...")
    mmdit.to(device)
    latent_sampled = do_sample(
        vae=vae,
        mmdit=mmdit,
        height=height,
        width=width,
        initial_latent=None,
        seed=seed,
        cond=cond,
        neg_cond=neg_cond,
        steps=steps,
        cfg_scale=cfg_scale,
        dtype=dtype,
        device=device,
    )
    if offload:
        mmdit.to("cpu")

    logger.info("Decoding...")
    vae.to(device)
    with torch.no_grad():
        image = vae.decode(latent_sampled)

    if offload:
        vae.to("cpu")

    image = image.float()
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
    decoded_np = (255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)).astype(np.uint8)
    out_image = Image.fromarray(decoded_np)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    out_image.save(output_path)
    logger.info(f"Saved image to {output_path}")


def parse_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.fp16:
        return torch.float16
    if args.bf16:
        return torch.bfloat16
    return torch.float32


def main():
    device = get_preferred_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--clip_g", type=str, required=False)
    parser.add_argument("--clip_l", type=str, required=False)
    parser.add_argument("--t5xxl", type=str, required=False)

    parser.add_argument("--prompt", type=str, default="A photo of a cat")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)

    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--offload", action="store_true", help="offload models to CPU between stages")

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument(
        "--mmdit_attn_output_gate",
        type=str,
        default="auto",
        choices=["auto", "none", "headwise", "elementwise"],
        help="gated attention after attn output: auto uses checkpoint if present",
    )
    parser.add_argument(
        "--mmdit_attn_output_gate_init_bias",
        type=float,
        default=2.0,
        help="initial bias for gate_proj when gate weights are not in ckpt; default 2.0",
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["torch", "xformers", "math"],
        help="attention backend for MMDiT",
    )

    args = parser.parse_args()

    sd3_dtype = parse_dtype(args)
    loading_device = "cpu" if args.offload else device

    logger.info(f"Loading SD3 checkpoint: {args.ckpt_path}")
    state_dict = load_safetensors(args.ckpt_path, loading_device, disable_mmap=True, dtype=sd3_dtype)

    clip_l = sd3_utils.load_clip_l(args.clip_l, sd3_dtype, loading_device, state_dict=state_dict)
    clip_g = sd3_utils.load_clip_g(args.clip_g, sd3_dtype, loading_device, state_dict=state_dict)
    t5xxl = sd3_utils.load_t5xxl(args.t5xxl, sd3_dtype, loading_device, state_dict=state_dict)

    vae = sd3_utils.load_vae(None, sd3_dtype, loading_device, state_dict=state_dict)

    attn_output_gate = None
    if args.mmdit_attn_output_gate == "none":
        attn_output_gate = None
    elif args.mmdit_attn_output_gate == "auto":
        attn_output_gate = None  # sd3_utils.load_mmdit will infer from state_dict if gate_proj exists
    else:
        attn_output_gate = args.mmdit_attn_output_gate

    mmdit = sd3_utils.load_mmdit(
        state_dict,
        sd3_dtype,
        loading_device,
        attn_mode=args.attn_mode,
        attn_output_gate=attn_output_gate,
        attn_output_gate_init_bias=args.mmdit_attn_output_gate_init_bias,
    )

    clip_l.eval()
    clip_g.eval()
    t5xxl.eval()
    mmdit.eval()
    vae.eval()

    if not args.offload:
        clip_l.to(device)
        clip_g.to(device)
        t5xxl.to(device)
        mmdit.to(device)
        vae.to(device)

    tokenize_strategy = strategy_sd3.Sd3TokenizeStrategy(256)
    encoding_strategy = strategy_sd3.Sd3TextEncodingStrategy()

    generate_image(
        mmdit=mmdit,
        vae=vae,
        clip_l=clip_l,
        clip_g=clip_g,
        t5xxl=t5xxl,
        tokenize_strategy=tokenize_strategy,
        encoding_strategy=encoding_strategy,
        steps=args.steps,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        width=args.width,
        height=args.height,
        cfg_scale=args.cfg_scale,
        dtype=sd3_dtype,
        device=device,
        offload=args.offload,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
