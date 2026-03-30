# Minimum Inference Code for Z-Image

import argparse
import math
import os
import random
import time
from typing import Any, Optional

import torch

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
    return prompts


def get_default_layerbind_hard_binding_layers(num_layers: int) -> list[int]:
    # Map the SD3.5 text-dominant layers onto the current model depth.
    sd35_reference = [0, 11, 14, 19, 21, 24, 29, 32, 34]
    sd35_max_index = 34
    mapped = {
        min(num_layers - 1, round(reference / sd35_max_index * max(num_layers - 1, 1)))
        for reference in sd35_reference
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
    states = []
    for region in layout.regions:
        indices = torch.tensor(region.token_indices, device=device, dtype=torch.long)
        keep_mask = torch.ones(x_seq_len, dtype=torch.bool, device=device)
        if indices.numel() > 0:
            keep_mask[indices] = False
        background_indices = all_indices[keep_mask]
        states.append(
            {
                "layer_index": region.layer_index,
                "bbox": region.bbox,
                "prompt": region.region_prompt,
                "indices": indices,
                "background_indices": background_indices,
                "branch_patches": None,
                "branch_tokens": None,
                "text_tokens": None,
                "alpha_mask": None,
            }
        )
    return states


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
    for order, region_state in enumerate(sorted_states):
        branch_tokens = region_state.get("branch_tokens")
        indices = region_state["indices"]
        if branch_tokens is None or indices.numel() == 0:
            continue

        current = blended.index_select(1, indices)
        if blend_mode == "direct" or order == 0:
            update = branch_tokens
            region_state["alpha_mask"] = torch.ones_like(branch_tokens[:, :, :1])
        else:
            alpha_mask = zimage_layerbind_utils.estimate_alpha_from_token_difference(
                branch_tokens,
                current,
                indices,
                token_shape=token_shape,
                gamma=gamma,
                poisson_lambda=poisson_lambda,
                beta=beta,
            )
            region_state["alpha_mask"] = alpha_mask
            update = current + alpha_mask * (branch_tokens - current)

        blended.index_copy_(1, indices, update)
    return blended


def compose_phase2_region_tokens(
    x_tokens: torch.Tensor,
    region_state: dict[str, Any],
    beta: float,
):
    branch_tokens = region_state.get("branch_tokens")
    indices = region_state["indices"]
    if branch_tokens is None or indices.numel() == 0:
        return x_tokens

    current = x_tokens.index_select(1, indices)
    update = current.lerp(branch_tokens, float(beta))
    composed = x_tokens.clone()
    composed.index_copy_(1, indices, update)
    return composed


def save_debug_image(vae, latents: torch.Tensor, file_path: str):
    latents = latents.to(vae.dtype)
    latents = zimage_train_utils._unscale_latents(latents, vae)
    decoded = zimage_train_utils._decode_latents(vae, latents)
    image = zimage_train_utils._latents_to_pil(decoded)
    image.save(file_path)


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
    phase2_beta_scale: float,
    phase2_delta_scale: float,
    phase2_branch_context_scale: float,
    apply_phase1_blend: bool,
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
                background_tokens, background_freqs = transformer.select_token_subset(
                    x_tokens, region_state["background_indices"], region_x_freqs
                )
                local_background_tokens = background_tokens
                local_background_freqs = background_freqs

                if layer_idx in hard_binding_layers:
                    region_state["branch_tokens"] = layer.contextual_forward(
                        region_state["branch_tokens"],
                        branch_freqs,
                        context_states=[region_state["text_tokens"]],
                        context_freqs_cis=[region_condition["freqs"]],
                        adaln_input=adaln_input,
                        include_query_in_kv=True,
                    )
                    if background_tokens.shape[1] > 0:
                        _, background_bg_freqs = transformer.select_token_subset(
                            x_tokens, region_state["background_indices"], x_freqs_cis
                        )
                        _, branch_bg_freqs = transformer.select_token_subset(
                            x_tokens, region_state["indices"], x_freqs_cis
                        )
                        adapted_background = layer.contextual_forward(
                            background_tokens,
                            background_bg_freqs,
                            context_states=[cap_tokens_current, region_state["branch_tokens"]],
                            context_freqs_cis=[cap_freqs_current, branch_bg_freqs],
                            adaln_input=adaln_input,
                            include_query_in_kv=True,
                        )
                        local_background_tokens = adapted_background
                        local_background_freqs = background_freqs
                else:
                    region_state["branch_tokens"] = layer.contextual_forward(
                        region_state["branch_tokens"],
                        branch_freqs,
                        context_states=[background_tokens, region_state["text_tokens"]],
                        context_freqs_cis=[background_freqs, region_condition["freqs"]],
                        adaln_input=adaln_input,
                        include_query_in_kv=True,
                    )
                region_state["alpha_mask"] = None

                region_state["text_tokens"] = layer.contextual_forward(
                    region_state["text_tokens"],
                    region_condition["freqs"],
                    context_states=[region_state["branch_tokens"], local_background_tokens],
                    context_freqs_cis=[branch_freqs, local_background_freqs],
                    adaln_input=adaln_input,
                    include_query_in_kv=True,
                )

        elif phase == "phase2":
            composed_x_tokens = x_tokens
            for region_state, region_condition in zip(region_states, region_conditions):
                if region_state["indices"].numel() == 0:
                    continue
                region_x_freqs = create_image_freqs_for_caption_length(
                    transformer,
                    x_meta["token_shape"],
                    cap_seq_len=region_condition["tokens"].shape[1],
                    batch_size=composed_x_tokens.shape[0],
                    device=composed_x_tokens.device,
                )
                region_tokens, region_freqs = transformer.select_token_subset(
                    composed_x_tokens, region_state["indices"], region_x_freqs
                )
                if region_state["text_tokens"] is None:
                    region_state["text_tokens"] = region_condition["tokens"].clone()
                branch_prior = region_state.get("branch_tokens")
                if branch_prior is None or branch_prior.shape != region_tokens.shape:
                    branch_prior = region_tokens
                branch_context = region_tokens.lerp(branch_prior, float(phase2_branch_context_scale))
                local_tokens = layer.contextual_forward(
                    region_tokens,
                    region_freqs,
                    context_states=[region_state["text_tokens"], branch_context, composed_x_tokens],
                    context_freqs_cis=[region_condition["freqs"], region_freqs, region_x_freqs],
                    adaln_input=adaln_input,
                    include_query_in_kv=True,
                )
                local_tokens = region_tokens.lerp(local_tokens, float(phase2_delta_scale))
                region_state["text_tokens"] = layer.contextual_forward(
                    region_state["text_tokens"],
                    region_condition["freqs"],
                    context_states=[local_tokens, cap_tokens_current],
                    context_freqs_cis=[region_freqs, cap_freqs_current],
                    adaln_input=adaln_input,
                    include_query_in_kv=True,
                )
                region_state["branch_tokens"] = local_tokens
                region_state["alpha_mask"] = None
                composed_x_tokens = compose_phase2_region_tokens(
                    composed_x_tokens,
                    region_state,
                    beta * float(phase2_beta_scale),
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
        if save_intermediates:
            intermediate_dir = os.path.join(output_dir, "layerbind_debug")
            os.makedirs(intermediate_dir, exist_ok=True)

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
                    phase2_beta_scale=layerbind_layout.config.phase2_beta_scale,
                    phase2_delta_scale=layerbind_layout.config.phase2_delta_scale,
                    phase2_branch_context_scale=layerbind_layout.config.phase2_branch_context_scale,
                    apply_phase1_blend=phase == "phase1" and (i + 1) == t1_step,
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

            if save_intermediates and intermediate_dir is not None and t1_step > 0 and (i + 1) == t1_step:
                save_debug_image(
                    vae,
                    latents,
                    os.path.join(intermediate_dir, f"{output_name or 'layerbind'}_t1_step_{i+1:02d}.png"),
                )

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
