import argparse
import os
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

import imageio
import soundfile as sf
import torch
from diffusers.utils import load_image
from diffusers.video_processor import VideoProcessor
from safetensors.torch import load_file
from tqdm import tqdm

from library.magi_utils import (
    MagiModelWrapper,
    MagiTrainInput,
    _str_to_torch_dtype,
    import_magi_components,
    load_magi_configs,
    load_magi_dit_model,
)
from library.utils import setup_logging
from networks import lora_magi

setup_logging()
import logging

logger = logging.getLogger(__name__)


@dataclass
class MagiInferenceContext:
    wrapper: MagiModelWrapper
    vae: object
    audio_vae: object
    model_cfg: object
    data_proxy_cfg: object
    get_padded_t5_gemma_embedding: object
    target_length: int
    device: torch.device
    model_dtype: torch.dtype
    decode_dtype: torch.dtype
    audio_in_channels: int


def _maybe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move_module(module_obj, device: torch.device, dtype: Optional[torch.dtype] = None) -> None:
    if hasattr(module_obj, "to"):
        if dtype is None:
            module_obj.to(device=device)
        else:
            module_obj.to(device=device, dtype=dtype)


def _move_audio_vae(audio_vae, device: torch.device) -> None:
    if hasattr(audio_vae, "vae_model"):
        audio_vae.vae_model.to(device)


def _maybe_offload_vae(ctx: MagiInferenceContext, enabled: bool) -> None:
    if enabled:
        _move_module(ctx.vae, torch.device("cpu"))
        _maybe_empty_cuda_cache()


def _maybe_offload_audio_vae(ctx: MagiInferenceContext, enabled: bool) -> None:
    if enabled:
        _move_audio_vae(ctx.audio_vae, torch.device("cpu"))
        _maybe_empty_cuda_cache()


def _get_text_embeddings(
    ctx: MagiInferenceContext,
    prompt: str,
    txt_model_path: str,
    offload_text_encoder: bool,
) -> Tuple[torch.Tensor, int]:
    prompt_embeds, prompt_len = ctx.get_padded_t5_gemma_embedding(
        prompt, txt_model_path, str(ctx.device), ctx.model_dtype, ctx.target_length
    )
    prompt_embeds = prompt_embeds.to(device=ctx.device, dtype=ctx.model_dtype).contiguous()
    prompt_len = int(prompt_len)

    if offload_text_encoder:
        from inference.model.t5_gemma.t5_gemma_model import get_t5_gemma_encoder

        encoder = get_t5_gemma_encoder(txt_model_path, str(ctx.device), ctx.model_dtype)
        if hasattr(encoder, "model"):
            encoder.model.to(torch.device("cpu"))
        _maybe_empty_cuda_cache()

    return prompt_embeds, prompt_len


def add_common_inference_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Path to daVinci DiT checkpoint.")
    parser.add_argument("--config_load_path", type=str, default=None, help="Optional daVinci config.json path.")
    parser.add_argument("--vae_model_path", type=str, required=True, help="Path to Wan2.2_VAE.pth.")
    parser.add_argument("--audio_model_path", type=str, required=True, help="Path to daVinci audio model.")
    parser.add_argument("--txt_model_path", type=str, required=True, help="Path to T5-Gemma encoder model.")

    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_path", type=str, default=None, help="Optional reference image path used as the first frame.")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument(
        "--audio_guidance_scale",
        type=float,
        default=None,
        help="Audio CFG scale. Defaults to --guidance_scale when not specified.",
    )
    parser.add_argument(
        "--low_t_video_guidance",
        type=float,
        default=2.0,
        help="Video CFG scale used when timestep <= 500 (matches daVinci default behavior).",
    )
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--discrete_flow_shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--width", type=int, default=480, help="Output video width in pixels.")
    parser.add_argument("--height", type=int, default=272, help="Output video height in pixels.")
    parser.add_argument("--num_frames", type=int, default=81, help="Output frame count.")
    parser.add_argument("--fps", type=int, default=24)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_dtype", type=str, default="bf16", help="bf16/fp16/fp32")
    parser.add_argument("--decode_dtype", type=str, default="bf16", help="bf16/fp16/fp32")
    parser.add_argument("--offload_text_encoder", action="store_true", help="Move T5-Gemma encoder back to CPU after prompt encoding.")
    parser.add_argument("--offload_vae", action="store_true", help="Keep video VAE on CPU except when encoding/decoding latents.")
    parser.add_argument("--offload_audio_vae", action="store_true", help="Keep audio VAE on CPU except when decoding audio latents.")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_name", type=str, default="magi_sample")
    parser.add_argument("--video_only", action="store_true", help="Save silent video only (skip native audio decode/mux).")
    return parser


def add_lora_inference_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--lora_weights",
        type=str,
        nargs="+",
        required=True,
        help="LoRA weights, each argument is `path` or `path;multiplier`.",
    )
    parser.add_argument("--merge_lora_weights", action="store_true", help="Merge LoRA weights into backbone.")
    return parser


def _parse_lora_spec(spec: str) -> Tuple[str, float]:
    if ";" not in spec:
        return spec, 1.0
    path, multiplier = spec.split(";", 1)
    return path, float(multiplier)


def _make_output_path(output_dir: str, output_name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return os.path.join(output_dir, f"{output_name}_{ts}.mp4")


def _write_video_mp4(video_frames, output_path: str, fps: int) -> None:
    # Keep close to daVinci pipeline defaults.
    imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])


def _to_thwc_uint8(video_tensor: torch.Tensor):
    if video_tensor.ndim != 4:
        raise ValueError(f"Expected decoded video tensor ndim=4, got {video_tensor.ndim}")

    if video_tensor.shape[0] in (1, 3):
        # C, T, H, W -> T, H, W, C
        video_tensor = video_tensor.permute(1, 2, 3, 0)
    elif video_tensor.shape[-1] in (1, 3):
        # already T, H, W, C
        pass
    else:
        raise ValueError(f"Unexpected decoded video shape: {tuple(video_tensor.shape)}")

    video_tensor = (video_tensor.float().clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1)
    return (video_tensor * 255.0).to(torch.uint8).cpu().numpy()


def _decode_audio_latent(audio_vae, latent_audio: torch.Tensor):
    from inference.pipeline.video_process import resample_audio_sinc

    latent_audio = latent_audio.squeeze(0)
    audio_output = audio_vae.decode(latent_audio.T)
    audio_output_np = audio_output.squeeze(0).T.cpu().numpy()
    # Keep consistent with daVinci post process.
    audio_output_np = resample_audio_sinc(audio_output_np, 441 / 512)
    return audio_output_np


def _decode_video_latent(vae, latent_video: torch.Tensor, decode_dtype: torch.dtype):
    try:
        from inference.infra.distributed import get_cp_group

        cp_group = get_cp_group()
    except Exception:
        cp_group = None

    decoded = vae.decode(latent_video.squeeze(0).to(decode_dtype), group=cp_group)
    if isinstance(decoded, (list, tuple)):
        video_tensor = decoded[0]
    elif isinstance(decoded, torch.Tensor):
        video_tensor = decoded[0] if decoded.ndim == 5 else decoded
    else:
        raise TypeError(f"Unsupported VAE decode output type: {type(decoded)}")
    return _to_thwc_uint8(video_tensor)


def _build_latent_shape(model_cfg, data_proxy_cfg, width: int, height: int, num_frames: int) -> Tuple[int, int, int, int, int]:
    # Keep aligned with daVinci evaluator defaults:
    # vae_stride=(4,16,16), patch=(t_patch_size, patch_size, patch_size).
    patch_t = int(data_proxy_cfg.t_patch_size)
    patch_hw = int(data_proxy_cfg.patch_size)
    patch_volume = patch_t * patch_hw * patch_hw

    video_in_channels = int(model_cfg.video_in_channels)
    if video_in_channels % patch_volume != 0:
        raise ValueError(
            f"Invalid channel/patch config: video_in_channels={video_in_channels}, patch_volume={patch_volume}. "
            "Expected divisibility for token<->latent conversion."
        )
    z_channels = video_in_channels // patch_volume

    latent_h = (int(height) // 16 // patch_hw) * patch_hw
    latent_w = (int(width) // 16 // patch_hw) * patch_hw
    if latent_h <= 0 or latent_w <= 0:
        raise ValueError(f"Invalid output size for latent build: width={width}, height={height}, patch={patch_hw}")

    latent_t = ((int(num_frames) - 1) // 4) + 1
    if patch_t > 1:
        latent_t = max(patch_t, (latent_t // patch_t) * patch_t)

    return (1, z_channels, latent_t, latent_h, latent_w)


def _encode_image_latent(ctx: MagiInferenceContext, image_path: str, latent_shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
    from inference.pipeline.video_process import resizecrop

    _, _, _, latent_h, latent_w = latent_shape
    image = load_image(image_path)
    target_height = int(latent_h * 16)
    target_width = int(latent_w * 16)
    image = resizecrop(image, target_height, target_width)
    image = VideoProcessor(vae_scale_factor=16).preprocess(image, height=target_height, width=target_width)
    image = image.to(device=ctx.device, dtype=ctx.decode_dtype).unsqueeze(2)
    image_latent = ctx.vae.encode(image).to(torch.float32)
    return image_latent


def _model_forward(
    ctx: MagiInferenceContext,
    noisy_video: torch.Tensor,
    noisy_audio: torch.Tensor,
    text_embeds: torch.Tensor,
    text_len: int,
):
    packed = MagiTrainInput(
        x_t=noisy_video,
        audio_x_t=noisy_audio,
        audio_feat_len=[int(noisy_audio.shape[1])],
        txt_feat=text_embeds,
        txt_feat_len=[int(text_len)],
    )
    model_inputs = ctx.wrapper.data_proxy.process_input(packed)
    pred_tokens = ctx.wrapper.model(*model_inputs)
    pred_video, pred_audio = ctx.wrapper.data_proxy.process_output(pred_tokens)
    return pred_video, pred_audio


def prepare_inference_context(args: argparse.Namespace) -> MagiInferenceContext:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Magi inference currently requires CUDA device because daVinci data proxy uses CUDA-only tensors.")
    model_dtype = _str_to_torch_dtype(args.model_dtype, default=torch.bfloat16)
    decode_dtype = _str_to_torch_dtype(args.decode_dtype, default=torch.bfloat16)

    model_cfg, data_proxy_cfg, t5_target_length = load_magi_configs(args.config_load_path)
    comps = import_magi_components()
    MagiDataProxy = comps["MagiDataProxy"]
    get_vae2_2 = comps["get_vae2_2"]
    get_padded_t5_gemma_embedding = comps["get_padded_t5_gemma_embedding"]
    from inference.model.sa_audio import SAAudioFeatureExtractor

    model = load_magi_dit_model(
        args.pretrained_model_name_or_path,
        model_cfg,
        device,
        model_dtype,
    )
    model.eval()

    data_proxy = MagiDataProxy(data_proxy_cfg)
    wrapper = MagiModelWrapper(model, data_proxy, audio_in_channels=model_cfg.audio_in_channels).to(device)
    wrapper.eval()

    logger.info(f"Loading Wan2.2 VAE from {args.vae_model_path}")
    vae = get_vae2_2(args.vae_model_path, device=str(device), weight_dtype=decode_dtype)
    vae = vae.to(device=device, dtype=decode_dtype)
    vae.vae.eval()

    logger.info(f"Loading audio VAE from {args.audio_model_path}")
    audio_vae = SAAudioFeatureExtractor(device=str(device), model_path=args.audio_model_path)

    _maybe_offload_vae_obj = bool(args.offload_vae)
    _maybe_offload_audio_obj = bool(args.offload_audio_vae)
    if _maybe_offload_vae_obj:
        _maybe_offload_vae(
            MagiInferenceContext(
                wrapper=wrapper,
                vae=vae,
                audio_vae=audio_vae,
                model_cfg=model_cfg,
                data_proxy_cfg=data_proxy_cfg,
                get_padded_t5_gemma_embedding=get_padded_t5_gemma_embedding,
                target_length=t5_target_length,
                device=device,
                model_dtype=model_dtype,
                decode_dtype=decode_dtype,
                audio_in_channels=model_cfg.audio_in_channels,
            ),
            True,
        )
    if _maybe_offload_audio_obj:
        _maybe_offload_audio_vae(
            MagiInferenceContext(
                wrapper=wrapper,
                vae=vae,
                audio_vae=audio_vae,
                model_cfg=model_cfg,
                data_proxy_cfg=data_proxy_cfg,
                get_padded_t5_gemma_embedding=get_padded_t5_gemma_embedding,
                target_length=t5_target_length,
                device=device,
                model_dtype=model_dtype,
                decode_dtype=decode_dtype,
                audio_in_channels=model_cfg.audio_in_channels,
            ),
            True,
        )

    return MagiInferenceContext(
        wrapper=wrapper,
        vae=vae,
        audio_vae=audio_vae,
        model_cfg=model_cfg,
        data_proxy_cfg=data_proxy_cfg,
        get_padded_t5_gemma_embedding=get_padded_t5_gemma_embedding,
        target_length=t5_target_length,
        device=device,
        model_dtype=model_dtype,
        decode_dtype=decode_dtype,
        audio_in_channels=model_cfg.audio_in_channels,
    )


def apply_lora_to_wrapper(wrapper: MagiModelWrapper, lora_specs: List[str], merge_lora_weights: bool) -> None:
    if len(lora_specs) == 0:
        raise ValueError("No LoRA weights provided.")
    if len(lora_specs) > 1:
        raise ValueError(
            "Current lora_magi inference path supports only a single LoRA weight. "
            "Please merge multiple LoRAs offline first."
        )

    model = wrapper.model
    loaded = 0
    for spec in lora_specs:
        weights_file, multiplier = _parse_lora_spec(spec)
        logger.info(f"Loading LoRA: {weights_file} (multiplier={multiplier})")
        weights_sd = load_file(weights_file)

        lora_net, _ = lora_magi.create_network_from_weights(
            multiplier=multiplier,
            file=None,
            vae=None,
            text_encoder=None,
            unet=model,
            weights_sd=weights_sd,
            for_inference=True,
        )

        if merge_lora_weights:
            lora_net.merge_to(None, model, weights_sd)
        else:
            lora_net.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
            info = lora_net.load_state_dict(weights_sd, strict=False)
            logger.info(f"Applied LoRA weights: {weights_file}, info={info}")
            lora_net.eval()
        loaded += 1

    logger.info(f"Loaded LoRA modules: {loaded}, merge={merge_lora_weights}")


@torch.no_grad()
def generate_video(
    ctx: MagiInferenceContext,
    args: argparse.Namespace,
):
    from inference.pipeline.video_process import merge_video_and_audio
    from inference.pipeline.scheduler_unipc import FlowUniPCMultistepScheduler

    prompt_embeds, prompt_len = _get_text_embeddings(
        ctx,
        args.prompt,
        args.txt_model_path,
        bool(args.offload_text_encoder),
    )

    audio_cfg_scale = float(args.audio_guidance_scale) if args.audio_guidance_scale is not None else float(args.guidance_scale)
    do_cfg = max(float(args.guidance_scale), audio_cfg_scale, float(args.low_t_video_guidance)) > 1.0

    if do_cfg:
        if args.negative_prompt.strip():
            neg_embeds, neg_len = _get_text_embeddings(
                ctx,
                args.negative_prompt,
                args.txt_model_path,
                bool(args.offload_text_encoder),
            )
        else:
            # Avoid T5-Gemma empty-string masking edge cases by treating empty negative prompts
            # as unconditional generation with zero text tokens.
            neg_embeds = torch.zeros_like(prompt_embeds)
            neg_len = 0
    else:
        neg_embeds = None
        neg_len = 0

    latent_shape = _build_latent_shape(ctx.model_cfg, ctx.data_proxy_cfg, args.width, args.height, args.num_frames)
    generator = torch.Generator(device=ctx.device).manual_seed(int(args.seed))
    latent_video = torch.randn(latent_shape, device=ctx.device, dtype=torch.float32, generator=generator)
    latent_audio = torch.randn(
        (1, int(args.num_frames), int(ctx.audio_in_channels)),
        device=ctx.device,
        dtype=torch.float32,
        generator=generator,
    )
    if args.image_path:
        if args.offload_vae:
            _move_module(ctx.vae, ctx.device, ctx.decode_dtype)
        latent_image = _encode_image_latent(ctx, args.image_path, latent_shape)
        _maybe_offload_vae(ctx, bool(args.offload_vae))
    else:
        latent_image = None

    video_scheduler = FlowUniPCMultistepScheduler()
    audio_scheduler = FlowUniPCMultistepScheduler()
    video_scheduler.set_timesteps(int(args.num_inference_steps), device=ctx.device, shift=float(args.discrete_flow_shift))
    audio_scheduler.set_timesteps(int(args.num_inference_steps), device=ctx.device, shift=float(args.discrete_flow_shift))
    timesteps = video_scheduler.timesteps
    logger.info(f"Start inference: steps={len(timesteps)}, latent_shape={tuple(latent_video.shape)}, cfg={args.guidance_scale}")
    for t in tqdm(timesteps, desc="magi_infer"):
        if latent_image is not None:
            latent_video[:, :, :1] = latent_image[:, :, :1]

        noisy_video = latent_video.to(dtype=ctx.model_dtype)
        noisy_audio = latent_audio.to(dtype=ctx.model_dtype)
        pred_cond_video, pred_cond_audio = _model_forward(
            ctx=ctx,
            noisy_video=noisy_video,
            noisy_audio=noisy_audio,
            text_embeds=prompt_embeds,
            text_len=prompt_len,
        )

        if neg_embeds is not None:
            pred_uncond_video, pred_uncond_audio = _model_forward(
                ctx=ctx,
                noisy_video=noisy_video,
                noisy_audio=noisy_audio,
                text_embeds=neg_embeds,
                text_len=neg_len,
            )
            t_val = float(t.item()) if hasattr(t, "item") else float(t)
            video_cfg_scale = float(args.guidance_scale) if t_val > 500.0 else float(args.low_t_video_guidance)
            pred_video = pred_uncond_video + video_cfg_scale * (pred_cond_video - pred_uncond_video)
            pred_audio = pred_uncond_audio + audio_cfg_scale * (pred_cond_audio - pred_uncond_audio)
        else:
            pred_video = pred_cond_video
            pred_audio = pred_cond_audio

        latent_video = video_scheduler.step(pred_video.to(torch.float32), t, latent_video, return_dict=False)[0]
        latent_audio = audio_scheduler.step(pred_audio.to(torch.float32), t, latent_audio, return_dict=False)[0]

    if latent_image is not None:
        latent_video[:, :, :1] = latent_image[:, :, :1]

    if args.offload_vae:
        _move_module(ctx.vae, ctx.device, ctx.decode_dtype)
    frames = _decode_video_latent(ctx.vae, latent_video, ctx.decode_dtype)
    _maybe_offload_vae(ctx, bool(args.offload_vae))
    out_path = _make_output_path(args.output_dir, args.output_name)

    if args.video_only:
        _write_video_mp4(frames, out_path, int(args.fps))
        logger.info(f"Saved silent video: {out_path}, frames={frames.shape[0]}, size={frames.shape[2]}x{frames.shape[1]}")
        return out_path

    # Native style: decode audio latents then mux with video.
    if args.offload_audio_vae:
        _move_audio_vae(ctx.audio_vae, ctx.device)
    audio_np = _decode_audio_latent(ctx.audio_vae, latent_audio)
    _maybe_offload_audio_vae(ctx, bool(args.offload_audio_vae))
    tmp_tag = uuid.uuid4().hex
    tmp_video = os.path.join(args.output_dir, f".{args.output_name}_{tmp_tag}.video.mp4")
    tmp_audio = os.path.join(args.output_dir, f".{args.output_name}_{tmp_tag}.audio.wav")

    try:
        _write_video_mp4(frames, tmp_video, int(args.fps))
        sf.write(tmp_audio, audio_np, ctx.audio_vae.sample_rate)
        merge_video_and_audio(tmp_video, tmp_audio, out_path)
    finally:
        try:
            os.remove(tmp_video)
        except OSError:
            pass
        try:
            os.remove(tmp_audio)
        except OSError:
            pass

    logger.info(
        f"Saved video+audio: {out_path}, frames={frames.shape[0]}, size={frames.shape[2]}x{frames.shape[1]}, "
        f"audio_sr={ctx.audio_vae.sample_rate}"
    )
    return out_path
