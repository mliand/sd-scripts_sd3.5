import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from library.magi_utils import (
    build_item_key,
    default_audio_cache_name,
    import_magi_components,
    load_jsonl_records,
    rebase_manifest_path,
    resolve_data_path,
    save_jsonl_records,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _pick_audio_path(record: dict, audio_field: str) -> str:
    candidates = [
        record.get(audio_field),
        record.get("audio"),
        record.get("audio_path"),
    ]
    for value in candidates:
        if value is not None and str(value).strip() != "":
            return str(value)
    raise ValueError(f"No audio path found in record: {record}")


def _resolve_seconds(record: dict, seconds: float | None, seconds_field: str | None) -> float | None:
    if seconds is not None:
        return float(seconds)
    if seconds_field:
        value = record.get(seconds_field)
        if value is not None and str(value).strip() != "":
            return float(value)
    return None


def cache_audio_latents(args: argparse.Namespace) -> None:
    records = load_jsonl_records(args.dataset_jsonl)
    os.makedirs(args.output_dir, exist_ok=True)

    import_magi_components()
    from inference.model.sa_audio import SAAudioFeatureExtractor
    from inference.pipeline.video_process import load_audio_and_encode

    device = torch.device(args.device)
    logger.info(f"Loading audio VAE from {args.audio_model_path}")
    audio_vae = SAAudioFeatureExtractor(device=str(device), model_path=args.audio_model_path)

    updated_records = []
    output_manifest = args.output_manifest
    if output_manifest is None:
        stem = Path(args.dataset_jsonl).stem
        output_manifest = os.path.join(args.output_dir, f"{stem}.with_audio_cache.jsonl")

    for idx, record in enumerate(tqdm(records, desc="Caching Magi audio latents")):
        item_key = build_item_key(record, idx)
        audio_value = _pick_audio_path(record, args.audio_field)
        audio_path = resolve_data_path(args.dataset_jsonl, audio_value)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        cache_name = default_audio_cache_name(item_key)
        cache_path = os.path.join(args.output_dir, cache_name)
        seconds = _resolve_seconds(record, args.seconds, args.seconds_field)

        if args.skip_existing and os.path.isfile(cache_path):
            logger.info(f"Skip existing audio cache: {cache_path}")
        else:
            audio_latents = load_audio_and_encode(audio_vae, audio_path, seconds)
            audio_latents = audio_latents.permute(0, 2, 1).squeeze(0).to(torch.float32).cpu().contiguous()
            tensors = {
                "audio_latents": audio_latents,
                "audio_len": torch.tensor([int(audio_latents.shape[0])], dtype=torch.int32),
                "audio_channels": torch.tensor([int(audio_latents.shape[1])], dtype=torch.int32),
            }
            metadata = {
                "architecture": "magi_human_audio",
                "format_version": "1.0.0",
                "item_key": item_key,
                "source_audio": audio_value,
            }
            if seconds is not None:
                metadata["seconds"] = str(seconds)
            save_file(tensors, cache_path, metadata=metadata)

        rel_cache = os.path.relpath(cache_path, os.path.dirname(os.path.abspath(output_manifest)))
        new_record = dict(record)
        if "magi_latent_cache" in new_record and new_record["magi_latent_cache"]:
            new_record["magi_latent_cache"] = rebase_manifest_path(
                args.dataset_jsonl, str(new_record["magi_latent_cache"]), output_manifest
            )
        if "magi_te_cache" in new_record and new_record["magi_te_cache"]:
            new_record["magi_te_cache"] = rebase_manifest_path(
                args.dataset_jsonl, str(new_record["magi_te_cache"]), output_manifest
            )
        new_record["magi_audio_cache"] = rel_cache
        updated_records.append(new_record)

    save_jsonl_records(output_manifest, updated_records)
    logger.info(f"Saved manifest with audio cache paths: {output_manifest}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache audio latents for daVinci-MagiHuman training in sd-scripts.")
    parser.add_argument("--dataset_jsonl", type=str, required=True, help="Input JSONL dataset manifest.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store audio cache files.")
    parser.add_argument("--output_manifest", type=str, default=None, help="Output JSONL with `magi_audio_cache` field.")
    parser.add_argument("--audio_model_path", type=str, required=True, help="Path to Stable Audio model directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for audio encoding.")
    parser.add_argument("--audio_field", type=str, default="audio", help="Preferred key name for audio path in JSONL.")
    parser.add_argument("--seconds", type=float, default=None, help="Optional fixed duration limit for audio encoding.")
    parser.add_argument("--seconds_field", type=str, default=None, help="Optional JSONL field containing per-item duration in seconds.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip audio cache files that already exist.")
    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    cache_audio_latents(args)


if __name__ == "__main__":
    main()
