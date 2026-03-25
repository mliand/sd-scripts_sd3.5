import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from library.magi_utils import (
    _str_to_torch_dtype,
    build_item_key,
    default_te_cache_name,
    get_caption,
    import_magi_components,
    load_jsonl_records,
    save_jsonl_records,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def cache_text_encoder_outputs(args: argparse.Namespace) -> None:
    records = load_jsonl_records(args.dataset_jsonl)
    os.makedirs(args.output_dir, exist_ok=True)

    comps = import_magi_components()
    get_padded_t5_gemma_embedding = comps["get_padded_t5_gemma_embedding"]

    device = torch.device(args.device)
    te_dtype = _str_to_torch_dtype(args.text_encoder_dtype, default=torch.bfloat16)

    updated_records = []
    output_manifest = args.output_manifest
    if output_manifest is None:
        stem = Path(args.dataset_jsonl).stem
        output_manifest = os.path.join(args.output_dir, f"{stem}.with_te_cache.jsonl")

    for idx, record in enumerate(tqdm(records, desc="Caching Magi text encoder outputs")):
        item_key = build_item_key(record, idx)
        caption = get_caption(record)

        cache_name = default_te_cache_name(item_key)
        cache_path = os.path.join(args.output_dir, cache_name)

        if args.skip_existing and os.path.isfile(cache_path):
            logger.info(f"Skip existing TE cache: {cache_path}")
        else:
            prompt_embeds, prompt_len = get_padded_t5_gemma_embedding(
                caption,
                args.txt_model_path,
                str(device),
                te_dtype,
                args.target_length,
            )
            # prompt_embeds: [1, L, D]
            prompt_embeds = prompt_embeds.squeeze(0).to(torch.float32).cpu().contiguous()
            tensors = {
                "prompt_embeds": prompt_embeds,
                "prompt_len": torch.tensor([int(prompt_len)], dtype=torch.int32),
                "target_length": torch.tensor([int(args.target_length)], dtype=torch.int32),
            }
            save_file(tensors, cache_path)

        rel_cache = os.path.relpath(cache_path, os.path.dirname(os.path.abspath(output_manifest)))
        new_record = dict(record)
        new_record["magi_te_cache"] = rel_cache
        updated_records.append(new_record)

    save_jsonl_records(output_manifest, updated_records)
    logger.info(f"Saved manifest with TE cache paths: {output_manifest}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache text encoder outputs for daVinci-MagiHuman training in sd-scripts.")
    parser.add_argument("--dataset_jsonl", type=str, required=True, help="Input JSONL dataset manifest.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store TE cache files.")
    parser.add_argument("--output_manifest", type=str, default=None, help="Output JSONL with `magi_te_cache` field.")

    parser.add_argument("--txt_model_path", type=str, required=True, help="Path to T5-Gemma encoder model.")
    parser.add_argument("--target_length", type=int, default=640, help="Pad/trim length for T5-Gemma hidden states.")

    parser.add_argument("--device", type=str, default="cuda", help="Device for text encoding.")
    parser.add_argument("--text_encoder_dtype", type=str, default="bf16", help="Text encoder dtype: bf16/fp16/fp32.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip TE files that already exist.")
    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    cache_text_encoder_outputs(args)


if __name__ == "__main__":
    main()
