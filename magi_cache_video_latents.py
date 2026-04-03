import argparse
import os
from pathlib import Path
from typing import List

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm

from library.magi_utils import (
    _str_to_torch_dtype,
    build_item_key,
    default_latent_cache_name,
    import_magi_components,
    load_jsonl_records,
    resolve_data_path,
    save_jsonl_records,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _pick_video_path(record: dict, video_field: str) -> str:
    candidates = [
        record.get(video_field),
        record.get("video"),
        record.get("video_path"),
        record.get("path"),
    ]
    for value in candidates:
        if value is not None and str(value).strip() != "":
            return str(value)
    raise ValueError(f"No video path found in record: {record}")


def _sample_frames(frames: List[np.ndarray], num_frames: int, frame_stride: int, sampling: str) -> List[np.ndarray]:
    if len(frames) == 0:
        raise ValueError("Video contains no frames.")

    if sampling == "head":
        sampled = frames[::frame_stride][:num_frames]
    elif sampling == "uniform":
        if len(frames) >= num_frames:
            idx = np.linspace(0, len(frames) - 1, num_frames, dtype=np.int64)
            sampled = [frames[int(i)] for i in idx]
        else:
            sampled = frames[:]
    else:
        raise ValueError(f"Unsupported sampling method: {sampling}")

    if len(sampled) < num_frames:
        sampled = sampled + [sampled[-1]] * (num_frames - len(sampled))

    return sampled


def _resize_to_bucket(image: Image.Image, width: int, height: int) -> Image.Image:
    src_w, src_h = image.size
    if src_w == width and src_h == height:
        return image

    scale = max(width / src_w, height / src_h)
    resized_w = int(src_w * scale + 0.5)
    resized_h = int(src_h * scale + 0.5)
    image = image.resize((resized_w, resized_h), Image.LANCZOS)

    left = max(0, (resized_w - width) // 2)
    top = max(0, (resized_h - height) // 2)
    return image.crop((left, top, left + width, top + height))


def _resize_frames(frames: List[np.ndarray], width: int, height: int, mode: str) -> List[np.ndarray]:
    out = []
    for frame in frames:
        image = Image.fromarray(frame)
        image = image.convert("RGB")
        if mode == "stretch":
            image = image.resize((width, height), Image.BICUBIC)
        elif mode == "bucket":
            image = _resize_to_bucket(image, width, height)
        else:
            raise ValueError(f"Unsupported resize mode: {mode}")
        out.append(np.array(image))
    return out


def _video_to_tensor(frames: List[np.ndarray], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    video = np.stack(frames, axis=0)  # [F, H, W, C]
    tensor = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # [1, C, F, H, W]
    tensor = tensor.to(device=device, dtype=dtype)
    tensor = tensor / 127.5 - 1.0
    return tensor


def _dtype_to_cache_str(dtype: torch.dtype) -> str:
    mapping = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
    }
    return mapping.get(dtype, str(dtype).replace("torch.", ""))


def cache_video_latents(args: argparse.Namespace) -> None:
    records = load_jsonl_records(args.dataset_jsonl)
    os.makedirs(args.output_dir, exist_ok=True)

    comps = import_magi_components()
    get_vae2_2 = comps["get_vae2_2"]

    device = torch.device(args.device)
    vae_dtype = _str_to_torch_dtype(args.vae_dtype, default=torch.bfloat16)

    logger.info(f"Loading Wan2.2 VAE from {args.vae_model_path}")
    vae = get_vae2_2(args.vae_model_path, device=str(device), weight_dtype=vae_dtype)
    vae = vae.to(device=device, dtype=vae_dtype)
    vae.vae.eval()

    updated_records = []
    output_manifest = args.output_manifest
    if output_manifest is None:
        stem = Path(args.dataset_jsonl).stem
        output_manifest = os.path.join(args.output_dir, f"{stem}.with_latents.jsonl")

    for idx, record in enumerate(tqdm(records, desc="Caching Magi video latents")):
        item_key = build_item_key(record, idx)
        video_value = _pick_video_path(record, args.video_field)
        video_path = resolve_data_path(args.dataset_jsonl, video_value)

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cache_name = default_latent_cache_name(item_key, args.num_frames, args.height, args.width)
        cache_path = os.path.join(args.output_dir, cache_name)

        if args.skip_existing and os.path.isfile(cache_path):
            logger.info(f"Skip existing latent cache: {cache_path}")
        else:
            source_frames = [f for f in iio.imiter(video_path)]
            sampled_frames = _sample_frames(source_frames, args.num_frames, args.frame_stride, args.sampling)

            src_h, src_w = sampled_frames[0].shape[:2]
            source_frame_count = len(source_frames)

            frames = _resize_frames(sampled_frames, args.width, args.height, args.resize_mode)

            video_tensor = _video_to_tensor(frames, device=device, dtype=vae_dtype)
            with torch.no_grad():
                latent = vae.encode(video_tensor).to(torch.float32)  # [1, C, T, H, W]

            latent_cpu = latent.squeeze(0).cpu().contiguous()
            _, latent_f, latent_h, latent_w = latent_cpu.shape
            latent_dtype = _dtype_to_cache_str(latent_cpu.dtype)
            musubi_style_key = f"latents_{latent_f}x{latent_h}x{latent_w}_{latent_dtype}"
            meta_tensors = {
                "latent_video": latent_cpu,
                musubi_style_key: latent_cpu,
                "frame_count": torch.tensor([args.num_frames], dtype=torch.int32),
                "height": torch.tensor([args.height], dtype=torch.int32),
                "width": torch.tensor([args.width], dtype=torch.int32),
                "source_frame_count": torch.tensor([source_frame_count], dtype=torch.int32),
                "original_height": torch.tensor([src_h], dtype=torch.int32),
                "original_width": torch.tensor([src_w], dtype=torch.int32),
                "frame_stride": torch.tensor([args.frame_stride], dtype=torch.int32),
            }
            metadata = {
                "architecture": "magi_human",
                "format_version": "1.0.1",
                "item_key": item_key,
                "source_video": video_value,
                "sampling": args.sampling,
                "resize_mode": args.resize_mode,
                "frame_count": str(args.num_frames),
                "width": str(args.width),
                "height": str(args.height),
                "source_frame_count": str(source_frame_count),
                "original_width": str(src_w),
                "original_height": str(src_h),
            }
            save_file(meta_tensors, cache_path, metadata=metadata)

        rel_cache = os.path.relpath(cache_path, os.path.dirname(os.path.abspath(output_manifest)))
        new_record = dict(record)
        new_record["magi_latent_cache"] = rel_cache
        updated_records.append(new_record)

    save_jsonl_records(output_manifest, updated_records)
    logger.info(f"Saved manifest with latent cache paths: {output_manifest}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache video latents for daVinci-MagiHuman training in sd-scripts.")
    parser.add_argument("--dataset_jsonl", type=str, required=True, help="Input JSONL dataset manifest.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store latent cache files.")
    parser.add_argument("--output_manifest", type=str, default=None, help="Output JSONL with `magi_latent_cache` field.")

    parser.add_argument("--vae_model_path", type=str, required=True, help="Path to Wan2.2_VAE.pth.")

    parser.add_argument("--device", type=str, default="cuda", help="Device for VAE encoding.")
    parser.add_argument("--vae_dtype", type=str, default="bf16", help="VAE dtype: bf16/fp16/fp32.")

    parser.add_argument("--video_field", type=str, default="video", help="Preferred key name for video path in JSONL.")
    parser.add_argument("--num_frames", type=int, default=256, help="Number of frames sampled per item.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Stride used in head sampling mode.")
    parser.add_argument("--sampling", type=str, default="head", choices=["head", "uniform"], help="Frame sampling strategy.")
    parser.add_argument(
        "--resize_mode",
        type=str,
        default="bucket",
        choices=["bucket", "stretch"],
        help="Frame resize mode: bucket keeps aspect ratio and center-crops, stretch does direct resize.",
    )
    parser.add_argument("--height", type=int, default=272, help="Resize height before VAE encoding.")
    parser.add_argument("--width", type=int, default=480, help="Resize width before VAE encoding.")

    parser.add_argument("--skip_existing", action="store_true", help="Skip latent files that already exist.")
    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    cache_video_latents(args)


if __name__ == "__main__":
    main()
