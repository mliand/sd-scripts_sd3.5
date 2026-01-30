import argparse
import sys
from typing import Dict

import torch
from safetensors.torch import load_file, save_file


REPLACE_RULES = (
    ("all_x_embedder.2-1.", "x_embedder."),
    ("all_final_layer.2-1.", "final_layer."),
    (".attention.to_out.0.", ".attention.out."),
    (".attention.norm_q.", ".attention.q_norm."),
    (".attention.norm_k.", ".attention.k_norm."),
)


def _apply_replacements(key: str) -> str:
    for src, dst in REPLACE_RULES:
        if src in key:
            key = key.replace(src, dst)
    return key


def convert_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    qkv_groups: Dict[str, Dict[str, torch.Tensor]] = {}

    for key, value in sd.items():
        # Merge Q/K/V into qkv
        if ".attention.to_q.weight" in key:
            base = key.replace(".attention.to_q.weight", ".attention")
            qkv_groups.setdefault(base, {})["q"] = value
            continue
        if ".attention.to_k.weight" in key:
            base = key.replace(".attention.to_k.weight", ".attention")
            qkv_groups.setdefault(base, {})["k"] = value
            continue
        if ".attention.to_v.weight" in key:
            base = key.replace(".attention.to_v.weight", ".attention")
            qkv_groups.setdefault(base, {})["v"] = value
            continue

        if ".attention.to_q.bias" in key:
            base = key.replace(".attention.to_q.bias", ".attention")
            qkv_groups.setdefault(base, {})["q_bias"] = value
            continue
        if ".attention.to_k.bias" in key:
            base = key.replace(".attention.to_k.bias", ".attention")
            qkv_groups.setdefault(base, {})["k_bias"] = value
            continue
        if ".attention.to_v.bias" in key:
            base = key.replace(".attention.to_v.bias", ".attention")
            qkv_groups.setdefault(base, {})["v_bias"] = value
            continue

        out[_apply_replacements(key)] = value

    for base, parts in qkv_groups.items():
        if {"q", "k", "v"} <= parts.keys():
            out[f"{base}.qkv.weight"] = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
        if {"q_bias", "k_bias", "v_bias"} <= parts.keys():
            out[f"{base}.qkv.bias"] = torch.cat([parts["q_bias"], parts["k_bias"], parts["v_bias"]], dim=0)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert sd-scripts Z-Image transformer weights to ComfyUI Lumina2/ZImage format"
    )
    parser.add_argument("src", help="Input transformer .safetensors (from sd-scripts)")
    parser.add_argument("dst", help="Output .safetensors for ComfyUI")
    args = parser.parse_args()

    sd = load_file(args.src)
    out = convert_state_dict(sd)
    if not out:
        print("No keys converted. Is this a transformer checkpoint?", file=sys.stderr)
        return 1

    save_file(out, args.dst)
    print(f"saved: {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
