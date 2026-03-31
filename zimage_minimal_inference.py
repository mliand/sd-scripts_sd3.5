# Minimum Inference Code for Z-Image

import argparse
import json
import math
import os
import random
import time
from typing import Any, Optional

import torch
from PIL import Image, ImageDraw

from library import strategy_zimage, train_util, zimage_layerbind_utils, zimage_utils
from library.device_utils import init_ipex, get_preferred_device
from library.utils import setup_logging
from library import zimage_train_utils

setup_logging()
import logging

logger = logging.getLogger(__name__)

init_ipex()


def parse_layer_spec(text: str):
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


def normalize_layer_indices(value):
    if value is None:
        return None
    if isinstance(value, str):
        return parse_layer_spec(value)
    if isinstance(value, list):
        return sorted(set(int(x) for x in value))
    return value


def add_layerbind_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--layerbind_layout", type=str, default=None, help="path to LayerBind layout JSON")
    parser.add_argument("--layerbind_eta1", type=float, default=0.20, help="Phase 1 step ratio for LayerBind")
    parser.add_argument("--layerbind_eta2", type=float, default=0.70, help="Phase 2 step ratio for LayerBind")
    parser.add_argument("--layerbind_beta", type=float, default=0.70, help="Layer transparency beta for LayerBind")
    parser.add_argument(
        "--layerbind_hard_binding_layers",
        type=str,
        default=None,
        help="Layer indices for LayerBind hard binding. Accepts the same syntax as --gate_layers.",
    )
    parser.add_argument(
        "--layerbind_blend_mode",
        type=str,
        default="alpha",
        choices=["direct", "alpha"],
        help="Region blending mode for LayerBind.",
    )
    parser.add_argument(
        "--layerbind_save_intermediates",
        action="store_true",
        help="save LayerBind intermediate images such as the t1 blend result",
    )
    parser.add_argument(
        "--layerbind_collect_layer_stats",
        action="store_true",
        help="collect Phase 1 CTA attention statistics to search Z-Image hard-binding layers",
    )
    parser.add_argument(
        "--layerbind_layer_stats_path",
        type=str,
        default=None,
        help="optional path to save LayerBind layer-search statistics JSON",
    )
    parser.add_argument(
        "--layerbind_layer_stats_top_k",
        type=int,
        default=None,
        help="number of hard-binding layers to recommend when saving layer-search statistics",
    )


def prepare_layerbind_layout(prompt_dict: dict[str, Any], width: int, height: int):
    layout_source = prompt_dict.get("layerbind_layout")
    if layout_source is None:
        layout_source = prompt_dict.get("layout")
    if layout_source is None:
        return None

    layout = zimage_layerbind_utils.load_layerbind_layout(layout_source)
    layout = zimage_layerbind_utils.populate_region_token_indices(layout, image_width=width, image_height=height)

    config = layout.config
    config.eta1 = float(prompt_dict.get("layerbind_eta1", config.eta1))
    config.eta2 = float(prompt_dict.get("layerbind_eta2", config.eta2))
    config.beta = float(prompt_dict.get("layerbind_beta", config.beta))
    layout.config = config
    return layout


def build_single_prompt_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "sample_steps": args.steps,
        "width": args.width,
        "height": args.height,
        "enum": 0,
        "layerbind_layout": args.layerbind_layout,
        "layerbind_eta1": args.layerbind_eta1,
        "layerbind_eta2": args.layerbind_eta2,
        "layerbind_beta": args.layerbind_beta,
        "layerbind_hard_binding_layers": args.layerbind_hard_binding_layers,
        "layerbind_blend_mode": args.layerbind_blend_mode,
        "layerbind_save_intermediates": args.layerbind_save_intermediates,
        "layerbind_collect_layer_stats": args.layerbind_collect_layer_stats,
        "layerbind_layer_stats_path": args.layerbind_layer_stats_path,
        "layerbind_layer_stats_top_k": args.layerbind_layer_stats_top_k,
    }


def build_prompts(args: argparse.Namespace):
    if args.sample_prompts is None:
        return [build_single_prompt_dict(args)]

    prompts = train_util.load_prompts(args.sample_prompts)
    for prompt in prompts:
        prompt.setdefault("layerbind_layout", args.layerbind_layout)
        prompt.setdefault("layerbind_eta1", args.layerbind_eta1)
        prompt.setdefault("layerbind_eta2", args.layerbind_eta2)
        prompt.setdefault("layerbind_beta", args.layerbind_beta)
        prompt.setdefault("layerbind_hard_binding_layers", args.layerbind_hard_binding_layers)
        prompt.setdefault("layerbind_blend_mode", args.layerbind_blend_mode)
        prompt.setdefault("layerbind_save_intermediates", args.layerbind_save_intermediates)
        prompt.setdefault("layerbind_collect_layer_stats", args.layerbind_collect_layer_stats)
        prompt.setdefault("layerbind_layer_stats_path", args.layerbind_layer_stats_path)
        prompt.setdefault("layerbind_layer_stats_top_k", args.layerbind_layer_stats_top_k)
    return prompts


def get_default_layerbind_hard_binding_layers(num_layers: int) -> list[int]:
    # Prefer the Z-Image 30-layer empirical layer-search result when depth matches the base model.
    zimage_reference = [0, 15, 16, 18, 19, 20, 27, 28, 29]
    if num_layers <= 0:
        return []
    if num_layers == 30:
        return zimage_reference

    zimage_max_index = 29
    mapped = {
        min(num_layers - 1, round(reference / zimage_max_index * max(num_layers - 1, 1)))
        for reference in zimage_reference
    }
    return sorted(mapped)


def resolve_layerbind_hard_binding_layers(prompt_dict: dict[str, Any], layout, transformer) -> list[int]:
    override = normalize_layer_indices(prompt_dict.get("layerbind_hard_binding_layers"))
    if override is not None:
        return override
    if layout is not None and layout.config.hard_binding_layers:
        return sorted(set(int(v) for v in layout.config.hard_binding_layers))
    return get_default_layerbind_hard_binding_layers(len(transformer.layers))


def get_image_sequence_length(transformer, width: int, height: int) -> int:
    patch_size = transformer.all_patch_size[0] if hasattr(transformer, "all_patch_size") else 2
    return (height // 8 // patch_size) * (width // 8 // patch_size)


def prepare_prompt_condition(
    transformer,
    tokenize_strategy,
    encoding_strategy,
    text_encoder,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    image_sequence_length: int,
):
    prompt_embeds, prompt_mask = zimage_train_utils._encode_prompt(
        tokenize_strategy,
        encoding_strategy,
        text_encoder,
        prompt,
        None,
        device,
        dtype,
    )
    prompt_embeds, prompt_mask = zimage_train_utils._trim_pad_embeds_and_mask(image_sequence_length, prompt_embeds, prompt_mask)
    cap_dtype = zimage_train_utils._get_model_param_dtype(transformer, dtype)
    prompt_embeds = prompt_embeds.to(dtype=cap_dtype)
    return {
        "prompt": prompt,
        "embeds": prompt_embeds,
        "mask": prompt_mask,
    }


def prepare_condition_tokens(transformer, condition: dict[str, Any]):
    tokens, freqs = transformer.prepare_caption_tokens(condition["embeds"], condition["mask"], apply_context_refiner=True)
    return {
        **condition,
        "tokens": tokens,
        "freqs": freqs,
    }


def prepare_layerbind_conditions(
    transformer,
    tokenize_strategy,
    encoding_strategy,
    text_encoder,
    prompt: str,
    negative_prompt: str,
    layout,
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
    do_cfg: bool,
):
    image_sequence_length = get_image_sequence_length(transformer, width, height)

    scene_prompt = layout.scene_prompt or prompt
    background_prompt = layout.background_prompt or scene_prompt
    scene_condition = prepare_condition_tokens(
        transformer,
        prepare_prompt_condition(
            transformer,
            tokenize_strategy,
            encoding_strategy,
            text_encoder,
            scene_prompt,
            device,
            dtype,
            image_sequence_length,
        ),
    )
    background_condition = prepare_condition_tokens(
        transformer,
        prepare_prompt_condition(
            transformer,
            tokenize_strategy,
            encoding_strategy,
            text_encoder,
            background_prompt,
            device,
            dtype,
            image_sequence_length,
        ),
    )
    region_conditions = []
    for region in layout.regions:
        condition = prepare_prompt_condition(
            transformer,
            tokenize_strategy,
            encoding_strategy,
            text_encoder,
            region.region_prompt,
            device,
            dtype,
            image_sequence_length,
        )
        region_conditions.append(prepare_condition_tokens(transformer, condition))

    negative_condition = None
    if do_cfg:
        negative_condition = prepare_prompt_condition(
            transformer,
            tokenize_strategy,
            encoding_strategy,
            text_encoder,
            negative_prompt,
            device,
            dtype,
            image_sequence_length,
        )

    return {
        "scene": scene_condition,
        "background": background_condition,
        "regions": region_conditions,
        "negative": negative_condition,
        "image_sequence_length": image_sequence_length,
    }


def prepare_region_runtime_states(layout, x_seq_len: int, device: torch.device):
    all_indices = torch.arange(x_seq_len, device=device, dtype=torch.long)
    all_region_mask = torch.zeros(x_seq_len, dtype=torch.bool, device=device)
    per_region_indices: list[torch.Tensor] = []
    for region in layout.regions:
        indices = torch.tensor(region.token_indices, device=device, dtype=torch.long)
        if indices.numel() > 0:
            all_region_mask[indices] = True
        per_region_indices.append(indices)

    states = []
    for region, indices in zip(layout.regions, per_region_indices):
        keep_mask = torch.ones(x_seq_len, dtype=torch.bool, device=device)
        if indices.numel() > 0:
            keep_mask[indices] = False
        background_indices = all_indices[keep_mask]
        is_occluding_hint = False
        for lower_region in layout.regions:
            if lower_region.layer_index >= region.layer_index:
                continue
            x1, y1, x2, y2 = region.bbox
            lx1, ly1, lx2, ly2 = lower_region.bbox
            if max(x1, lx1) < min(x2, lx2) and max(y1, ly1) < min(y2, ly2):
                is_occluding_hint = True
                break
        foreign_region_mask = all_region_mask.clone()
        if indices.numel() > 0:
            foreign_region_mask[indices] = False
        foreign_region_indices = all_indices[foreign_region_mask]
        region_mask = torch.ones((1, indices.numel(), 1), device=device, dtype=torch.float32)
        states.append(
            {
                "layer_index": region.layer_index,
                "bbox": region.bbox,
                "prompt": region.region_prompt,
                "indices": indices,
                "background_indices": background_indices,
                "foreign_region_indices": foreign_region_indices,
                "is_occluding_hint": is_occluding_hint,
                "branch_patches": None,
                "branch_tokens": None,
                "text_tokens": None,
                "region_mask": region_mask,
                "alpha_mask": None,
            }
        )
    return states


def build_layerbind_local_context_indices(
    region_indices: torch.Tensor,
    token_shape: tuple[int, int, int],
    seq_len: int,
    device: torch.device,
    forbidden_indices: Optional[torch.Tensor] = None,
    radius: int = 8,
    global_anchor_count: int = 32,
) -> torch.Tensor:
    if region_indices.numel() == 0:
        return torch.zeros((0,), device=device, dtype=torch.long)

    _, token_height, token_width = token_shape
    if token_height * token_width != seq_len:
        all_indices = torch.arange(seq_len, device=device, dtype=torch.long)
        region_mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
        region_mask[region_indices] = True
        return all_indices[~region_mask]

    region_y = torch.div(region_indices, token_width, rounding_mode="floor")
    region_x = region_indices % token_width
    x1 = max(0, int(region_x.min().item()) - int(radius))
    y1 = max(0, int(region_y.min().item()) - int(radius))
    x2 = min(token_width, int(region_x.max().item()) + int(radius) + 1)
    y2 = min(token_height, int(region_y.max().item()) + int(radius) + 1)

    y_coords = torch.arange(y1, y2, device=device, dtype=torch.long)
    x_coords = torch.arange(x1, x2, device=device, dtype=torch.long)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    local_rect_indices = (yy * token_width + xx).reshape(-1)

    all_indices = torch.arange(seq_len, device=device, dtype=torch.long)
    region_mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
    region_mask[region_indices] = True
    forbidden_mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
    if forbidden_indices is not None and forbidden_indices.numel() > 0:
        forbidden_mask[forbidden_indices] = True
    local_mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
    local_mask[local_rect_indices] = True

    local_background = all_indices[local_mask & ~region_mask & ~forbidden_mask]
    global_candidates = all_indices[~local_mask & ~region_mask & ~forbidden_mask]

    if global_anchor_count > 0 and global_candidates.numel() > 0:
        if global_candidates.numel() <= global_anchor_count:
            anchors = global_candidates
        else:
            sample_positions = torch.linspace(
                0,
                global_candidates.numel() - 1,
                steps=global_anchor_count,
                device=device,
            ).round()
            anchors = global_candidates.index_select(0, sample_positions.to(dtype=torch.long))
    else:
        anchors = torch.zeros((0,), device=device, dtype=torch.long)

    context_indices = torch.cat([local_background, anchors], dim=0)
    if context_indices.numel() == 0:
        context_indices = all_indices[~region_mask & ~forbidden_mask]
    if context_indices.numel() == 0:
        context_indices = all_indices[~region_mask]
    return torch.unique(context_indices, sorted=True)


def build_layerbind_segment_logit_biases(
    include_query_in_kv: bool,
    query_length: int,
    context_lengths: list[int],
    context_roles: list[str],
) -> list[float]:
    if len(context_lengths) != len(context_roles):
        raise ValueError(f"context length/role mismatch: {len(context_lengths)} vs {len(context_roles)}")

    biases: list[float] = []
    if include_query_in_kv:
        biases.append(-math.log(max(int(query_length), 1)))

    for length, role in zip(context_lengths, context_roles):
        bias = -math.log(max(int(length), 1))
        if role == "text":
            bias += 1.0
        elif role == "scene_text":
            bias += 0.2
        elif role == "branch":
            bias += 0.2
        elif role == "local_global":
            bias += 0.0
        biases.append(bias)

    return biases


def create_image_freqs_for_caption_length(
    transformer,
    token_shape: tuple[int, int, int],
    cap_seq_len: int,
    batch_size: int,
    device: torch.device,
):
    position_ids = transformer.create_image_position_ids(
        token_shape[0], token_shape[1], token_shape[2], cap_seq_len=cap_seq_len, device=device
    )
    freqs_cis = transformer.rope_embedder(position_ids)
    return freqs_cis.unsqueeze(0).expand(batch_size, -1, -1)


def prepare_branch_image_tokens(
    transformer,
    branch_patches: torch.Tensor,
    branch_freqs: torch.Tensor,
    adaln_input: torch.Tensor,
    patch_size: int,
    f_patch_size: int,
):
    embedder = transformer.all_x_embedder[f"{patch_size}-{f_patch_size}"]
    branch_tokens = embedder(branch_patches.to(dtype=embedder.weight.dtype))
    adaln_input = adaln_input.type_as(branch_tokens)

    if len(transformer.noise_refiner) > 0:
        noise_refiner_attn_params = transformer.create_main_attention_params(0, None)
        for layer in transformer.noise_refiner:
            branch_tokens = layer(branch_tokens, branch_freqs, adaln_input, attn_params=noise_refiner_attn_params)

    return branch_tokens


def predict_branch_patch_residual(
    transformer,
    branch_tokens: torch.Tensor,
    adaln_input: torch.Tensor,
    patch_size: int,
    f_patch_size: int,
):
    final_layer = transformer.all_final_layer[f"{patch_size}-{f_patch_size}"]
    return final_layer(branch_tokens, adaln_input.type_as(branch_tokens))


def create_layerbind_layer_stats_accumulator(transformer, prompt_dict: dict[str, Any], layout) -> Optional[dict[str, Any]]:
    if layout is None or not bool(prompt_dict.get("layerbind_collect_layer_stats", False)):
        return None

    return {
        "num_layers": len(transformer.layers),
        "top_k": prompt_dict.get("layerbind_layer_stats_top_k"),
        "layout_scene_prompt": layout.scene_prompt,
        "layout_background_prompt": layout.background_prompt,
        "regions": [
            {
                "layer_index": region.layer_index,
                "bbox": list(region.bbox),
                "token_count": len(region.token_indices),
                "region_prompt": region.region_prompt,
            }
            for region in layout.regions
        ],
        "layers": {
            str(layer_idx): {
                "self_attention_sum": 0.0,
                "background_attention_sum": 0.0,
                "text_attention_sum": 0.0,
                "query_vector_count": 0.0,
            }
            for layer_idx in range(len(transformer.layers))
        },
    }


def record_layerbind_layer_stats(
    accumulator: Optional[dict[str, Any]],
    layer_idx: int,
    stats: dict[str, float],
):
    if accumulator is None:
        return

    layer_stats = accumulator["layers"][str(layer_idx)]
    query_vector_count = float(stats.get("query_vector_count", 0.0))
    if query_vector_count <= 0.0:
        return

    layer_stats["self_attention_sum"] += float(stats.get("segment_attention/self", 0.0)) * query_vector_count
    layer_stats["background_attention_sum"] += float(stats.get("segment_attention/background", 0.0)) * query_vector_count
    layer_stats["text_attention_sum"] += float(stats.get("segment_attention/text", 0.0)) * query_vector_count
    layer_stats["query_vector_count"] += query_vector_count


def finalize_layerbind_layer_stats(accumulator: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if accumulator is None:
        return None

    num_layers = int(accumulator["num_layers"])
    default_top_k = len(get_default_layerbind_hard_binding_layers(num_layers))
    top_k = accumulator.get("top_k")
    if top_k is None:
        top_k = default_top_k
    top_k = max(1, min(int(top_k), num_layers))

    per_layer = []
    for layer_idx in range(num_layers):
        raw_stats = accumulator["layers"][str(layer_idx)]
        query_vector_count = float(raw_stats["query_vector_count"])
        if query_vector_count > 0.0:
            self_attention = raw_stats["self_attention_sum"] / query_vector_count
            background_attention = raw_stats["background_attention_sum"] / query_vector_count
            text_attention = raw_stats["text_attention_sum"] / query_vector_count
        else:
            self_attention = 0.0
            background_attention = 0.0
            text_attention = 0.0

        per_layer.append(
            {
                "layer_idx": layer_idx,
                "query_vector_count": query_vector_count,
                "self_attention": self_attention,
                "background_attention": background_attention,
                "text_attention": text_attention,
                "text_minus_background": text_attention - background_attention,
                "text_over_background": text_attention / max(background_attention, 1e-6),
            }
        )

    ranked_layers = sorted(
        [item for item in per_layer if item["query_vector_count"] > 0.0],
        key=lambda item: (item["text_minus_background"], item["text_over_background"], -item["layer_idx"]),
        reverse=True,
    )
    if not ranked_layers:
        suggested_layers = get_default_layerbind_hard_binding_layers(num_layers)
    else:
        suggested_layers = [0]
        for item in ranked_layers:
            layer_idx = int(item["layer_idx"])
            if layer_idx == 0:
                continue
            suggested_layers.append(layer_idx)
            if len(suggested_layers) >= top_k:
                break
        suggested_layers = sorted(set(suggested_layers))

    return {
        "num_layers": num_layers,
        "top_k": top_k,
        "suggested_hard_binding_layers": suggested_layers,
        "default_mapped_layers": get_default_layerbind_hard_binding_layers(num_layers),
        "layout_scene_prompt": accumulator.get("layout_scene_prompt", ""),
        "layout_background_prompt": accumulator.get("layout_background_prompt", ""),
        "regions": accumulator.get("regions", []),
        "layers": per_layer,
    }


def resolve_layerbind_layer_stats_output_path(
    prompt_dict: dict[str, Any],
    output_dir: str,
    output_name: Optional[str],
    seed: int,
    sample_steps: int,
    index: int,
):
    custom_path = prompt_dict.get("layerbind_layer_stats_path")
    if custom_path:
        return custom_path

    os.makedirs(output_dir, exist_ok=True)
    base_name = output_name or "layerbind"
    return os.path.join(output_dir, f"{base_name}_{sample_steps:06d}_{index:02d}_{seed}_layer_stats.json")


def blend_region_tokens(
    x_tokens: torch.Tensor,
    region_states: list[dict[str, Any]],
    beta: float,
    blend_mode: str,
    token_shape: tuple[int, int, int],
    gamma: float,
    poisson_lambda: float,
):
    blended = x_tokens.clone()
    sorted_states = sorted(region_states, key=lambda item: item["layer_index"])
    occupied = torch.zeros(x_tokens.shape[1], device=x_tokens.device, dtype=torch.bool)
    occluding_flags: list[bool] = []
    for region_state in sorted_states:
        indices = region_state["indices"]
        if indices.numel() == 0:
            occluding_flags.append(False)
            continue
        has_overlap = bool(occupied.index_select(0, indices).any().item())
        occluding_flags.append(has_overlap)
        occupied.index_fill_(0, indices, True)

    for region_state, is_occluding in zip(sorted_states, occluding_flags):
        branch_tokens = region_state.get("branch_tokens")
        indices = region_state["indices"]
        if branch_tokens is None or indices.numel() == 0:
            continue

        current = blended.index_select(1, indices)
        is_occluding = bool(region_state.get("is_occluding_hint", False)) or is_occluding
        region_state["is_occluding"] = is_occluding
        if blend_mode == "direct" or not is_occluding:
            if blend_mode == "direct":
                region_state["region_mask"] = torch.ones_like(branch_tokens[:, :, :1])
            else:
                _alpha_mask, binary_mask = zimage_layerbind_utils.estimate_alpha_from_token_difference(
                    branch_tokens,
                    current,
                    indices,
                    token_shape=token_shape,
                    gamma=gamma,
                    poisson_lambda=poisson_lambda,
                    return_binary_mask=True,
                )
                region_state["region_mask"] = binary_mask
            update = branch_tokens
            region_state["alpha_mask"] = None
        else:
            alpha_mask, binary_mask = zimage_layerbind_utils.estimate_alpha_from_token_difference(
                branch_tokens,
                current,
                indices,
                token_shape=token_shape,
                gamma=gamma,
                poisson_lambda=poisson_lambda,
                return_binary_mask=True,
            )
            region_state["region_mask"] = binary_mask
            region_state["alpha_mask"] = alpha_mask
            update = alpha_mask * branch_tokens + (1.0 - alpha_mask) * current

        blended.index_copy_(1, indices, update)
    return blended


def compose_phase2_region_tokens(
    x_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    indices: torch.Tensor,
    beta: float,
    region_mask: Optional[torch.Tensor] = None,
):
    if local_tokens is None or indices.numel() == 0:
        return x_tokens

    current = x_tokens.index_select(1, indices)
    if region_mask is not None and region_mask.shape[1] == current.shape[1]:
        # Layer transparency scheduler: alpha_o = beta * M.
        mask = region_mask.to(dtype=current.dtype)
        if mask.shape[0] == 1 and current.shape[0] > 1:
            mask = mask.expand(current.shape[0], -1, -1)
        if mask.shape[0] != current.shape[0]:
            mask = None
    else:
        mask = None

    if mask is not None:
        update = current + (float(beta) * mask) * (local_tokens - current)
    else:
        update = current.lerp(local_tokens, float(beta))
    composed = x_tokens.clone()
    composed.index_copy_(1, indices, update)
    return composed


def save_debug_image(vae, latents: torch.Tensor, file_path: str):
    latents = latents.to(vae.dtype)
    latents = zimage_train_utils._unscale_latents(latents, vae)
    decoded = zimage_train_utils._decode_latents(vae, latents)
    image = zimage_train_utils._latents_to_pil(decoded)
    image.save(file_path)


def save_boxed_debug_image(image: Image.Image, layout, file_path: str):
    boxed = image.copy()
    draw = ImageDraw.Draw(boxed)
    palette = [
        (255, 64, 64),
        (64, 192, 255),
        (64, 220, 96),
        (255, 176, 64),
        (192, 96, 255),
    ]

    for index, region in enumerate(layout.regions):
        color = palette[index % len(palette)]
        x1, y1, x2, y2 = region.bbox
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        label = f"{region.layer_index}:{region.region_prompt}"
        draw.text((x1 + 6, max(0, y1 + 6)), label, fill=color)

    boxed.save(file_path)


def get_layerbind_debug_save_points(sample_steps: int, percents: tuple[int, ...] = (10, 30, 60, 80)) -> list[tuple[int, int]]:
    if sample_steps <= 0:
        return []

    save_points = []
    for percent in percents:
        step = min(sample_steps, max(1, math.ceil(sample_steps * (percent / 100.0))))
        save_points.append((step, percent))

    deduped = {}
    for step, percent in save_points:
        deduped.setdefault(step, percent)
    return sorted((step, percent) for step, percent in deduped.items())


def run_layerbind_forward(
    transformer,
    latent_model_input: torch.Tensor,
    timestep: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
    scene_condition: dict[str, Any],
    background_condition: dict[str, Any],
    region_conditions: list[dict[str, Any]],
    region_states: list[dict[str, Any]],
    phase: str,
    hard_binding_layers: list[int],
    beta: float,
    blend_mode: str,
    gamma: float,
    poisson_lambda: float,
    phase2_delta_scale: float,
    apply_phase1_blend: bool,
    layer_stats_accumulator: Optional[dict[str, Any]] = None,
):
    active_condition = background_condition if phase == "phase1" else scene_condition
    cap_tokens = active_condition["tokens"]
    cap_mask = active_condition["mask"]
    cap_freqs = active_condition["freqs"]
    cap_seq_len = cap_tokens.shape[1]

    adaln_input = transformer.prepare_adaln_input(timestep)
    x_tokens, x_freqs_cis, x_meta = transformer.prepare_image_tokens(
        latent_model_input,
        cap_seq_len=cap_seq_len,
        patch_size=transformer.all_patch_size[0],
        f_patch_size=transformer.all_f_patch_size[0],
        adaln_input=adaln_input,
    )
    adaln_input = adaln_input.type_as(x_tokens)
    image_patches = None
    if phase == "phase1":
        image_patches = transformer.patchify(latent_model_input.to(torch.float32), x_meta["patch_size"], x_meta["f_patch_size"])

    cap_tokens_current = cap_tokens.clone()
    cap_freqs_current = cap_freqs
    unified_freqs_cis = torch.cat([x_freqs_cis, cap_freqs_current], dim=1)
    attn_params = transformer.create_main_attention_params(x_meta["seq_len"], cap_mask)
    for region_state in region_states:
        cached_seq_len = region_state.get("context_seq_len")
        if cached_seq_len != x_meta["seq_len"] or region_state.get("context_indices") is None:
            region_state["context_indices"] = build_layerbind_local_context_indices(
                region_state["indices"],
                token_shape=x_meta["token_shape"],
                seq_len=x_meta["seq_len"],
                device=x_tokens.device,
                forbidden_indices=region_state.get("foreign_region_indices"),
                radius=10,
                global_anchor_count=64,
            )
            region_state["context_seq_len"] = x_meta["seq_len"]

    if phase == "phase1":
        for region_state, region_condition in zip(region_states, region_conditions):
            if region_state["indices"].numel() == 0:
                continue
            region_x_freqs = create_image_freqs_for_caption_length(
                transformer,
                x_meta["token_shape"],
                cap_seq_len=region_condition["tokens"].shape[1],
                batch_size=x_tokens.shape[0],
                device=x_tokens.device,
            )
            branch_seed = image_patches.index_select(1, region_state["indices"])
            branch_freqs = region_x_freqs.index_select(1, region_state["indices"])
            if region_state["branch_patches"] is None or region_state["branch_patches"].shape != branch_seed.shape:
                # Phase 1 starts from the same latent noise patches as the global path, then evolves independently.
                region_state["branch_patches"] = branch_seed.clone()
            region_state["branch_tokens"] = prepare_branch_image_tokens(
                transformer,
                region_state["branch_patches"],
                branch_freqs,
                adaln_input,
                patch_size=x_meta["patch_size"],
                f_patch_size=x_meta["f_patch_size"],
            )
            if region_state["text_tokens"] is None:
                region_state["text_tokens"] = region_condition["tokens"].clone()

    for layer_idx, layer in enumerate(transformer.layers):
        unified, _ = transformer.build_unified_tokens(x_tokens, x_freqs_cis, cap_tokens_current, cap_freqs_current)
        unified = layer(unified, unified_freqs_cis, adaln_input, attn_params=attn_params)
        x_tokens, cap_tokens_current = transformer.split_unified_tokens(unified, x_meta["seq_len"])

        if phase == "phase1":
            for region_state, region_condition in zip(region_states, region_conditions):
                if region_state["indices"].numel() == 0:
                    continue

                region_x_freqs = create_image_freqs_for_caption_length(
                    transformer,
                    x_meta["token_shape"],
                    cap_seq_len=region_condition["tokens"].shape[1],
                    batch_size=x_tokens.shape[0],
                    device=x_tokens.device,
                )
                _, branch_freqs = transformer.select_token_subset(x_tokens, region_state["indices"], region_x_freqs)
                local_context_indices = region_state.get("context_indices", region_state["background_indices"])
                background_tokens, background_freqs = transformer.select_token_subset(
                    x_tokens, local_context_indices, region_x_freqs
                )
                local_background_tokens = background_tokens
                local_background_freqs = background_freqs
                include_query_in_kv = layer_idx in hard_binding_layers

                if layer_stats_accumulator is not None:
                    segment_names = ["self", "background", "text"] if include_query_in_kv else ["background", "text"]
                    layer_attention_stats = layer.contextual_attention_stats(
                        region_state["branch_tokens"],
                        branch_freqs,
                        context_states=[background_tokens, region_state["text_tokens"]],
                        context_freqs_cis=[background_freqs, region_condition["freqs"]],
                        adaln_input=adaln_input,
                        include_query_in_kv=include_query_in_kv,
                        segment_names=segment_names,
                    )
                    record_layerbind_layer_stats(layer_stats_accumulator, layer_idx, layer_attention_stats)

                if include_query_in_kv:
                    branch_segment_biases = build_layerbind_segment_logit_biases(
                        include_query_in_kv=True,
                        query_length=region_state["branch_tokens"].shape[1],
                        context_lengths=[region_state["text_tokens"].shape[1]],
                        context_roles=["text"],
                    )
                    region_state["branch_tokens"] = layer.contextual_forward(
                        region_state["branch_tokens"],
                        branch_freqs,
                        context_states=[region_state["text_tokens"]],
                        context_freqs_cis=[region_condition["freqs"]],
                        adaln_input=adaln_input,
                        include_query_in_kv=True,
                        segment_logit_biases=branch_segment_biases,
                    )
                    if background_tokens.shape[1] > 0:
                        _, background_bg_freqs = transformer.select_token_subset(
                            x_tokens, local_context_indices, x_freqs_cis
                        )
                        _, branch_bg_freqs = transformer.select_token_subset(
                            x_tokens, region_state["indices"], x_freqs_cis
                        )
                        background_segment_biases = build_layerbind_segment_logit_biases(
                            include_query_in_kv=True,
                            query_length=background_tokens.shape[1],
                            context_lengths=[cap_tokens_current.shape[1], region_state["branch_tokens"].shape[1]],
                            context_roles=["scene_text", "branch"],
                        )
                        adapted_background = layer.contextual_forward(
                            background_tokens,
                            background_bg_freqs,
                            context_states=[cap_tokens_current, region_state["branch_tokens"]],
                            context_freqs_cis=[cap_freqs_current, branch_bg_freqs],
                            adaln_input=adaln_input,
                            include_query_in_kv=True,
                            segment_logit_biases=background_segment_biases,
                        )
                        local_background_tokens = adapted_background
                        local_background_freqs = background_freqs
                else:
                    branch_segment_biases = build_layerbind_segment_logit_biases(
                        include_query_in_kv=False,
                        query_length=region_state["branch_tokens"].shape[1],
                        context_lengths=[background_tokens.shape[1], region_state["text_tokens"].shape[1]],
                        context_roles=["local_global", "text"],
                    )
                    region_state["branch_tokens"] = layer.contextual_forward(
                        region_state["branch_tokens"],
                        branch_freqs,
                        context_states=[background_tokens, region_state["text_tokens"]],
                        context_freqs_cis=[background_freqs, region_condition["freqs"]],
                        adaln_input=adaln_input,
                        include_query_in_kv=False,
                        segment_logit_biases=branch_segment_biases,
                    )
                region_state["alpha_mask"] = None

                text_include_query = include_query_in_kv
                text_segment_biases = build_layerbind_segment_logit_biases(
                    include_query_in_kv=text_include_query,
                    query_length=region_state["text_tokens"].shape[1],
                    context_lengths=[region_state["branch_tokens"].shape[1], local_background_tokens.shape[1]],
                    context_roles=["branch", "local_global"],
                )
                region_state["text_tokens"] = layer.contextual_forward(
                    region_state["text_tokens"],
                    region_condition["freqs"],
                    context_states=[region_state["branch_tokens"], local_background_tokens],
                    context_freqs_cis=[branch_freqs, local_background_freqs],
                    adaln_input=adaln_input,
                    include_query_in_kv=text_include_query,
                    segment_logit_biases=text_segment_biases,
                )

        elif phase == "phase2":
            global_x_tokens = x_tokens
            composed_x_tokens = global_x_tokens
            for region_state, region_condition in zip(region_states, region_conditions):
                if region_state["indices"].numel() == 0:
                    continue
                region_x_freqs = create_image_freqs_for_caption_length(
                    transformer,
                    x_meta["token_shape"],
                    cap_seq_len=region_condition["tokens"].shape[1],
                    batch_size=global_x_tokens.shape[0],
                    device=global_x_tokens.device,
                )
                region_tokens, region_freqs = transformer.select_token_subset(
                    global_x_tokens, region_state["indices"], region_x_freqs
                )
                local_context_indices = region_state.get("context_indices", region_state["background_indices"])
                local_global_tokens, local_global_freqs = transformer.select_token_subset(
                    global_x_tokens,
                    local_context_indices,
                    region_x_freqs,
                )
                if region_state["text_tokens"] is None:
                    region_state["text_tokens"] = region_condition["tokens"].clone()
                include_query_in_kv = layer_idx in hard_binding_layers
                local_segment_biases = build_layerbind_segment_logit_biases(
                    include_query_in_kv=include_query_in_kv,
                    query_length=region_tokens.shape[1],
                    context_lengths=[
                        region_state["text_tokens"].shape[1],
                        local_global_tokens.shape[1],
                    ],
                    context_roles=["text", "local_global"],
                )
                local_tokens = layer.contextual_forward(
                    region_tokens,
                    region_freqs,
                    context_states=[region_state["text_tokens"], local_global_tokens],
                    context_freqs_cis=[region_condition["freqs"], local_global_freqs],
                    adaln_input=adaln_input,
                    include_query_in_kv=include_query_in_kv,
                    segment_logit_biases=local_segment_biases,
                )
                region_injection_scale = 1.0 if region_state.get("is_occluding", False) else 0.60
                local_tokens = region_tokens.lerp(local_tokens, float(phase2_delta_scale) * region_injection_scale)
                text_include_query = False
                text_segment_biases = build_layerbind_segment_logit_biases(
                    include_query_in_kv=text_include_query,
                    query_length=region_state["text_tokens"].shape[1],
                    context_lengths=[local_tokens.shape[1]],
                    context_roles=["branch"],
                )
                region_state["text_tokens"] = layer.contextual_forward(
                    region_state["text_tokens"],
                    region_condition["freqs"],
                    context_states=[local_tokens],
                    context_freqs_cis=[region_freqs],
                    adaln_input=adaln_input,
                    include_query_in_kv=text_include_query,
                    segment_logit_biases=text_segment_biases,
                )
                region_state["branch_tokens"] = local_tokens
                composed_x_tokens = compose_phase2_region_tokens(
                    composed_x_tokens,
                    local_tokens,
                    region_state["indices"],
                    beta * region_injection_scale,
                    region_mask=region_state.get("region_mask"),
                )
            x_tokens = composed_x_tokens

    if phase == "phase1" and apply_phase1_blend:
        x_tokens = blend_region_tokens(
            x_tokens,
            region_states,
            beta,
            blend_mode,
            token_shape=x_meta["token_shape"],
            gamma=gamma,
            poisson_lambda=poisson_lambda,
        )
    elif phase == "phase1" and sigmas is not None:
        for region_state in region_states:
            branch_tokens = region_state.get("branch_tokens")
            branch_patches = region_state.get("branch_patches")
            if branch_tokens is None or branch_patches is None:
                continue
            branch_patch_residual = predict_branch_patch_residual(
                transformer,
                branch_tokens,
                adaln_input,
                patch_size=x_meta["patch_size"],
                f_patch_size=x_meta["f_patch_size"],
            )
            region_state["branch_patches"] = zimage_train_utils._step(
                (-branch_patch_residual).to(torch.float32),
                branch_patches.to(torch.float32),
                sigmas,
                step_index,
            )

    unified, _ = transformer.build_unified_tokens(x_tokens, x_freqs_cis, cap_tokens_current, cap_freqs_current)
    return transformer.finalize_image_tokens(
        unified,
        adaln_input,
        x_meta["image_shape"],
        patch_size=x_meta["patch_size"],
        f_patch_size=x_meta["f_patch_size"],
    )


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
    logger.info(f"seed: {seed}")

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

    layerbind_layout = prepare_layerbind_layout(prompt_dict, width, height)
    if layerbind_layout is not None:
        if layerbind_layout.scene_prompt:
            prompt = layerbind_layout.scene_prompt
        if layerbind_layout.negative_prompt:
            negative_prompt = layerbind_layout.negative_prompt

        logger.info(
            "layerbind layout loaded: regions=%s eta1=%.3f eta2=%.3f beta=%.3f",
            len(layerbind_layout.regions),
            layerbind_layout.config.eta1,
            layerbind_layout.config.eta2,
            layerbind_layout.config.beta,
        )
        for region in layerbind_layout.regions:
            logger.info(
                "layerbind region layer=%s bbox=%s tokens=%s prompt=%s",
                region.layer_index,
                region.bbox,
                len(region.token_indices),
                region.region_prompt,
            )
        if not hasattr(transformer, "prepare_image_tokens"):
            logger.warning("layerbind layout was provided but the loaded transformer does not expose LayerBind helpers; falling back to base inference")

    do_cfg = guidance_scale is not None and guidance_scale > 1.0
    use_layerbind = layerbind_layout is not None and hasattr(transformer, "prepare_image_tokens")
    image_sequence_length = get_image_sequence_length(transformer, width, height)
    prompt_condition = prepare_prompt_condition(
        transformer,
        tokenize_strategy,
        encoding_strategy,
        text_encoder,
        prompt,
        device,
        dtype,
        image_sequence_length,
    )
    prompt_embeds = prompt_condition["embeds"]
    prompt_mask = prompt_condition["mask"]

    negative_embeds = None
    negative_mask = None
    if do_cfg:
        negative_condition = prepare_prompt_condition(
            transformer,
            tokenize_strategy,
            encoding_strategy,
            text_encoder,
            negative_prompt,
            device,
            dtype,
            image_sequence_length,
        )
        negative_embeds = negative_condition["embeds"]
        negative_mask = negative_condition["mask"]

    channels = getattr(transformer, "in_channels", 16)
    latents = torch.randn(
        (1, channels, height // 8, width // 8),
        device=device,
        dtype=torch.float32,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    timesteps, sigmas = zimage_train_utils._get_timesteps_sigmas(sample_steps, discrete_flow_shift)
    timesteps = timesteps.to(device)
    sigmas = sigmas.to(device)

    layerbind_conditions = None
    hard_binding_layers = []
    t1_step = 0
    t2_step = 0
    blend_mode = prompt_dict.get("layerbind_blend_mode", "alpha")
    save_intermediates = bool(prompt_dict.get("layerbind_save_intermediates", False))
    intermediate_dir = None
    debug_save_points = []
    layer_stats_accumulator = None
    if use_layerbind:
        with torch.autocast(device_type=device.type, dtype=dtype), torch.no_grad():
            layerbind_conditions = prepare_layerbind_conditions(
                transformer,
                tokenize_strategy,
                encoding_strategy,
                text_encoder,
                prompt,
                negative_prompt,
                layerbind_layout,
                width,
                height,
                device,
                dtype,
                do_cfg,
            )
        prompt_embeds = layerbind_conditions["scene"]["embeds"]
        prompt_mask = layerbind_conditions["scene"]["mask"]
        if do_cfg and layerbind_conditions["negative"] is not None:
            negative_embeds = layerbind_conditions["negative"]["embeds"]
            negative_mask = layerbind_conditions["negative"]["mask"]

        hard_binding_layers = resolve_layerbind_hard_binding_layers(prompt_dict, layerbind_layout, transformer)
        t1_step = min(sample_steps, max(0, math.ceil(sample_steps * layerbind_layout.config.eta1)))
        t2_step = min(sample_steps, max(t1_step, math.ceil(sample_steps * layerbind_layout.config.eta2)))
        logger.info(
            "layerbind schedule: t1_step=%s t2_step=%s hard_binding_layers=%s blend_mode=%s",
            t1_step,
            t2_step,
            hard_binding_layers,
            blend_mode,
        )
        layer_stats_accumulator = create_layerbind_layer_stats_accumulator(transformer, prompt_dict, layerbind_layout)
        if save_intermediates:
            intermediate_dir = os.path.join(output_dir, "layerbind_debug")
            os.makedirs(intermediate_dir, exist_ok=True)
            debug_save_points = get_layerbind_debug_save_points(sample_steps)

    with torch.autocast(device_type=device.type, dtype=dtype), torch.no_grad():
        region_states = None
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000

            latent_model_input = latents.to(dtype).unsqueeze(2)
            if use_layerbind and i < t2_step:
                if region_states is None:
                    region_states = prepare_region_runtime_states(
                        layerbind_layout,
                        layerbind_conditions["image_sequence_length"],
                        device=latent_model_input.device,
                    )
                phase = "phase1" if i < t1_step else "phase2"
                model_out = run_layerbind_forward(
                    transformer,
                    latent_model_input,
                    timestep,
                    sigmas,
                    i,
                    layerbind_conditions["scene"],
                    layerbind_conditions["background"],
                    layerbind_conditions["regions"],
                    region_states,
                    phase=phase,
                    hard_binding_layers=hard_binding_layers,
                    beta=layerbind_layout.config.beta,
                    blend_mode=blend_mode,
                    gamma=layerbind_layout.config.gamma,
                    poisson_lambda=layerbind_layout.config.poisson_lambda,
                    phase2_delta_scale=layerbind_layout.config.phase2_delta_scale,
                    apply_phase1_blend=phase == "phase1" and (i + 1) == t1_step,
                    layer_stats_accumulator=layer_stats_accumulator,
                )
            else:
                model_out = transformer(x=latent_model_input, t=timestep, cap_feats=prompt_embeds, cap_mask=prompt_mask)

            if do_cfg:
                neg_out = transformer(x=latent_model_input, t=timestep, cap_feats=negative_embeds, cap_mask=negative_mask)
                noise_pred = model_out + guidance_scale * (model_out - neg_out)
            else:
                noise_pred = model_out

            noise_pred = -noise_pred.squeeze(2)
            latents = zimage_train_utils._step(noise_pred.to(torch.float32), latents, sigmas, i)

            if save_intermediates and intermediate_dir is not None:
                step_number = i + 1
                for save_step, save_percent in debug_save_points:
                    if step_number == save_step:
                        save_debug_image(
                            vae,
                            latents,
                            os.path.join(
                                intermediate_dir,
                                f"{output_name or 'layerbind'}_{save_percent:02d}pct_step_{step_number:02d}.png",
                            ),
                        )
                        break

        latents = latents.to(vae.dtype)
        latents = zimage_train_utils._unscale_latents(latents, vae)
        decoded = zimage_train_utils._decode_latents(vae, latents)
        image = zimage_train_utils._latents_to_pil(decoded)

    os.makedirs(output_dir, exist_ok=True)
    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"{sample_steps:06d}"
    seed_suffix = f"_{seed}"
    index = prompt_dict.get("enum", 0)
    filename = f"{'' if output_name is None else output_name + '_'}{num_suffix}_{index:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(output_dir, filename))
    if save_intermediates and intermediate_dir is not None and layerbind_layout is not None:
        save_boxed_debug_image(
            image,
            layerbind_layout,
            os.path.join(intermediate_dir, f"{output_name or 'layerbind'}_100pct_boxed.png"),
        )

    layer_stats_summary = finalize_layerbind_layer_stats(layer_stats_accumulator)
    if layer_stats_summary is not None:
        layer_stats_path = resolve_layerbind_layer_stats_output_path(
            prompt_dict,
            output_dir=output_dir,
            output_name=output_name,
            seed=seed,
            sample_steps=sample_steps,
            index=index,
        )
        with open(layer_stats_path, "w", encoding="utf-8") as handle:
            json.dump(layer_stats_summary, handle, indent=2, ensure_ascii=False)
        logger.info(
            "layerbind layer stats saved: %s suggested_hard_binding_layers=%s",
            layer_stats_path,
            layer_stats_summary["suggested_hard_binding_layers"],
        )


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
    add_layerbind_arguments(parser)
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

    args.layerbind_hard_binding_layers = normalize_layer_indices(args.layerbind_hard_binding_layers)
    args.gate_layers = normalize_layer_indices(args.gate_layers)
    args.gate_layers_noise_refiner = normalize_layer_indices(args.gate_layers_noise_refiner)
    args.gate_layers_context_refiner = normalize_layer_indices(args.gate_layers_context_refiner)

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

    prompts = build_prompts(args)

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
