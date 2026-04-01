import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from library import zimage_config


@dataclass
class LayerBindConfig:
    eta1: float = 0.20
    eta2: float = 0.70
    beta: float = 0.70
    gamma: float = 0.90
    poisson_lambda: float = 0.50
    phase2_delta_scale: float = 0.50
    hard_binding_layers: list[int] = field(default_factory=list)


@dataclass
class RegionLayer:
    region_prompt: str
    bbox: tuple[int, int, int, int]
    layer_index: int
    token_indices: list[int] = field(default_factory=list)


@dataclass
class LayerBindLayout:
    background_prompt: str = ""
    scene_prompt: str = ""
    negative_prompt: str = ""
    config: LayerBindConfig = field(default_factory=LayerBindConfig)
    regions: list[RegionLayer] = field(default_factory=list)


def _coerce_bbox(bbox: Sequence[Any]) -> tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements, got {bbox}")
    return tuple(int(v) for v in bbox)


def layerbind_config_from_dict(data: Optional[dict[str, Any]]) -> LayerBindConfig:
    if data is None:
        return LayerBindConfig()

    config = LayerBindConfig()
    for key in (
        "eta1",
        "eta2",
        "beta",
        "gamma",
        "poisson_lambda",
        "phase2_delta_scale",
    ):
        if key in data and data[key] is not None:
            setattr(config, key, float(data[key]))

    hard_binding_layers = data.get("hard_binding_layers")
    if hard_binding_layers is not None:
        config.hard_binding_layers = sorted(set(int(v) for v in hard_binding_layers))

    return config


def layerbind_layout_from_dict(data: dict[str, Any]) -> LayerBindLayout:
    regions_data = data.get("regions")
    if regions_data is None:
        regions_data = data.get("elements", [])

    regions = []
    for index, region in enumerate(regions_data, start=1):
        layer_index = region.get("layer_index", region.get("order", index))
        region_prompt = region.get("region_prompt", region.get("prompt", ""))
        bbox = _coerce_bbox(region.get("bbox", region.get("layout", (0, 0, 0, 0))))
        token_indices = [int(v) for v in region.get("token_indices", [])]
        regions.append(
            RegionLayer(
                region_prompt=region_prompt,
                bbox=bbox,
                layer_index=int(layer_index),
                token_indices=token_indices,
            )
        )

    regions = sorted(regions, key=lambda item: item.layer_index)
    return LayerBindLayout(
        background_prompt=str(data.get("background_prompt", "")),
        scene_prompt=str(data.get("scene_prompt", data.get("rewritten_prompt", ""))),
        negative_prompt=str(data.get("negative_prompt", "")),
        config=layerbind_config_from_dict(data.get("config", data)),
        regions=regions,
    )


def load_layerbind_layout(path_or_data: str | Path | dict[str, Any]) -> LayerBindLayout:
    if isinstance(path_or_data, dict):
        return layerbind_layout_from_dict(path_or_data)

    path = Path(path_or_data)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return layerbind_layout_from_dict(data)


def get_token_grid_size(
    image_width: int,
    image_height: int,
    vae_scale_factor: int = zimage_config.ZIMAGE_VAE_SCALE_FACTOR,
    patch_size: int = zimage_config.DEFAULT_TRANSFORMER_PATCH_SIZE[0],
) -> tuple[int, int]:
    latent_width = image_width // vae_scale_factor
    latent_height = image_height // vae_scale_factor
    token_width = latent_width // patch_size
    token_height = latent_height // patch_size
    return token_height, token_width


def bbox_to_token_bounds(
    bbox: Sequence[int],
    image_width: int,
    image_height: int,
    vae_scale_factor: int = zimage_config.ZIMAGE_VAE_SCALE_FACTOR,
    patch_size: int = zimage_config.DEFAULT_TRANSFORMER_PATCH_SIZE[0],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _coerce_bbox(bbox)
    token_height, token_width = get_token_grid_size(image_width, image_height, vae_scale_factor, patch_size)
    pixel_per_token = vae_scale_factor * patch_size

    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    tx1 = x1 // pixel_per_token
    ty1 = y1 // pixel_per_token
    tx2 = math.ceil(x2 / pixel_per_token)
    ty2 = math.ceil(y2 / pixel_per_token)

    tx1 = max(0, min(tx1, token_width))
    ty1 = max(0, min(ty1, token_height))
    tx2 = max(tx1, min(tx2, token_width))
    ty2 = max(ty1, min(ty2, token_height))

    return tx1, ty1, tx2, ty2


def token_bounds_to_indices(bounds: Sequence[int], token_width: int) -> list[int]:
    x1, y1, x2, y2 = _coerce_bbox(bounds)
    indices = []
    for y in range(y1, y2):
        row_offset = y * token_width
        for x in range(x1, x2):
            indices.append(row_offset + x)
    return indices


def bbox_to_token_indices(
    bbox: Sequence[int],
    image_width: int,
    image_height: int,
    vae_scale_factor: int = zimage_config.ZIMAGE_VAE_SCALE_FACTOR,
    patch_size: int = zimage_config.DEFAULT_TRANSFORMER_PATCH_SIZE[0],
) -> list[int]:
    token_height, token_width = get_token_grid_size(image_width, image_height, vae_scale_factor, patch_size)
    bounds = bbox_to_token_bounds(bbox, image_width, image_height, vae_scale_factor, patch_size)
    indices = token_bounds_to_indices(bounds, token_width)
    max_index = token_height * token_width
    return [index for index in indices if 0 <= index < max_index]


def token_indices_to_mask(
    token_indices: Sequence[int],
    image_width: int,
    image_height: int,
    vae_scale_factor: int = zimage_config.ZIMAGE_VAE_SCALE_FACTOR,
    patch_size: int = zimage_config.DEFAULT_TRANSFORMER_PATCH_SIZE[0],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    token_height, token_width = get_token_grid_size(image_width, image_height, vae_scale_factor, patch_size)
    mask = torch.zeros(token_height * token_width, dtype=torch.bool, device=device)
    if token_indices:
        indices = torch.tensor(sorted(set(int(v) for v in token_indices)), dtype=torch.long, device=mask.device)
        indices = indices[(indices >= 0) & (indices < mask.numel())]
        if indices.numel() > 0:
            mask.index_fill_(0, indices, True)
    return mask.view(token_height, token_width)


def _scatter_token_values(
    token_values: torch.Tensor,
    token_indices: Sequence[int] | torch.Tensor,
    token_shape: tuple[int, int, int],
) -> torch.Tensor:
    _, token_height, token_width = token_shape
    if isinstance(token_indices, torch.Tensor):
        indices = token_indices.to(device=token_values.device, dtype=torch.long)
    else:
        indices = torch.tensor(list(token_indices), device=token_values.device, dtype=torch.long)

    grid = torch.zeros(
        (token_values.shape[0], 1, token_height * token_width),
        device=token_values.device,
        dtype=token_values.dtype,
    )
    if indices.numel() > 0:
        scatter_index = indices.view(1, 1, -1).expand(token_values.shape[0], 1, -1)
        grid.scatter_(2, scatter_index, token_values.unsqueeze(1))
    return grid.view(token_values.shape[0], 1, token_height, token_width)


def _scatter_token_mask(
    token_indices: Sequence[int] | torch.Tensor,
    token_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    _, token_height, token_width = token_shape
    if isinstance(token_indices, torch.Tensor):
        indices = token_indices.to(device=device, dtype=torch.long)
    else:
        indices = torch.tensor(list(token_indices), device=device, dtype=torch.long)

    mask = torch.zeros((1, 1, token_height * token_width), device=device, dtype=torch.float32)
    if indices.numel() > 0:
        scatter_index = indices.view(1, 1, -1)
        mask.scatter_(2, scatter_index, torch.ones_like(scatter_index, dtype=torch.float32))
    return mask.view(1, 1, token_height, token_width)


def _screened_poisson_smooth(score_map: torch.Tensor, poisson_lambda: float, num_iters: int = 24) -> torch.Tensor:
    smoothed = F.avg_pool2d(score_map, kernel_size=3, stride=1, padding=1)
    for _ in range(num_iters):
        neighbors = (
            F.pad(smoothed[:, :, 1:, :], (0, 0, 0, 1))
            + F.pad(smoothed[:, :, :-1, :], (0, 0, 1, 0))
            + F.pad(smoothed[:, :, :, 1:], (0, 1, 0, 0))
            + F.pad(smoothed[:, :, :, :-1], (1, 0, 0, 0))
        )
        smoothed = (neighbors + float(poisson_lambda) * score_map) / (4.0 + float(poisson_lambda))
    return smoothed


def _otsu_threshold(values: torch.Tensor, num_bins: int = 64) -> float:
    if values.numel() == 0:
        return 0.0

    values = values.detach().float()
    v_min = values.min().item()
    v_max = values.max().item()
    if not math.isfinite(v_min) or not math.isfinite(v_max) or abs(v_max - v_min) < 1e-6:
        return float(v_min)

    hist = torch.histc(values.cpu(), bins=num_bins, min=v_min, max=v_max)
    prob = hist / hist.sum().clamp(min=1e-6)
    bin_centers = torch.linspace(v_min, v_max, steps=num_bins)
    omega = torch.cumsum(prob, dim=0)
    mu = torch.cumsum(prob * bin_centers, dim=0)
    mu_total = mu[-1]
    sigma_between = (mu_total * omega - mu).pow(2) / (omega * (1.0 - omega)).clamp(min=1e-6)
    return float(bin_centers[int(torch.argmax(sigma_between).item())].item())


def _binary_dilate(mask: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    out = mask.float()
    for _ in range(iterations):
        out = F.max_pool2d(out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (out > 0).float()


def _binary_erode(mask: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    out = mask.float()
    for _ in range(iterations):
        out = 1.0 - F.max_pool2d(1.0 - out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (out > 0).float()


def _morphology_refine(mask: torch.Tensor) -> torch.Tensor:
    closed = _binary_erode(_binary_dilate(mask, iterations=1), iterations=1)
    opened = _binary_dilate(_binary_erode(closed, iterations=1), iterations=1)
    return _binary_dilate(opened, iterations=1)


def _morphological_reconstruct(seed: torch.Tensor, mask: torch.Tensor, max_iters: int = 32) -> torch.Tensor:
    seed = (seed > 0).float()
    mask = (mask > 0).float()
    current = seed * mask
    for _ in range(max_iters):
        expanded = _binary_dilate(current, iterations=1) * mask
        if torch.equal((expanded > 0), (current > 0)):
            break
        current = expanded
    return current


def _connected_components_2d(mask_2d: torch.Tensor) -> list[torch.Tensor]:
    mask_cpu = (mask_2d > 0).to(device="cpu", dtype=torch.bool)
    height, width = mask_cpu.shape
    visited = torch.zeros((height, width), dtype=torch.bool)
    components: list[torch.Tensor] = []
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for y in range(height):
        for x in range(width):
            if not mask_cpu[y, x].item() or visited[y, x].item():
                continue
            component = torch.zeros((height, width), dtype=torch.float32)
            queue = deque([(y, x)])
            visited[y, x] = True
            component[y, x] = 1.0
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in neighbors:
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx].item() or not mask_cpu[ny, nx].item():
                        continue
                    visited[ny, nx] = True
                    component[ny, nx] = 1.0
                    queue.append((ny, nx))
            components.append(component.to(device=mask_2d.device))
    return components


def _build_region_core_mask(
    token_indices: Sequence[int] | torch.Tensor,
    token_shape: tuple[int, int, int],
    device: torch.device,
    core_ratio: float = 0.5,
) -> torch.Tensor:
    _, token_height, token_width = token_shape
    if isinstance(token_indices, torch.Tensor):
        indices = token_indices.to(device=device, dtype=torch.long)
    else:
        indices = torch.tensor(list(token_indices), device=device, dtype=torch.long)

    mask = torch.zeros((1, 1, token_height, token_width), device=device, dtype=torch.float32)
    if indices.numel() == 0:
        return mask

    y = torch.div(indices, token_width, rounding_mode="floor")
    x = indices.remainder(token_width)
    x1 = int(x.min().item())
    x2 = int(x.max().item()) + 1
    y1 = int(y.min().item())
    y2 = int(y.max().item()) + 1

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    core_ratio = float(max(0.2, min(core_ratio, 1.0)))
    core_width = max(1, round(width * core_ratio))
    core_height = max(1, round(height * core_ratio))
    core_x1 = x1 + max(0, (width - core_width) // 2)
    core_y1 = y1 + max(0, (height - core_height) // 2)
    core_x2 = min(x2, core_x1 + core_width)
    core_y2 = min(y2, core_y1 + core_height)
    mask[:, :, core_y1:core_y2, core_x1:core_x2] = 1.0
    return mask


def refine_alpha_mask_with_region_core(
    alpha_map: torch.Tensor,
    binary_mask: torch.Tensor,
    token_indices: Sequence[int] | torch.Tensor,
    token_shape: tuple[int, int, int],
    core_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    core_mask = _build_region_core_mask(token_indices, token_shape, binary_mask.device, core_ratio=core_ratio)
    refined_binary = torch.zeros_like(binary_mask)
    refined_alpha = torch.zeros_like(alpha_map)

    for batch_index in range(binary_mask.shape[0]):
        batch_binary = binary_mask[batch_index, 0]
        batch_alpha = alpha_map[batch_index, 0]
        batch_core = core_mask[0, 0]

        candidate_mask = batch_binary
        if (candidate_mask * batch_core).amax().item() <= 0:
            candidate_mask = batch_binary * _binary_dilate(core_mask, iterations=1)[0, 0]
        if (candidate_mask * batch_core).amax().item() <= 0:
            candidate_mask = batch_binary * _binary_dilate(core_mask, iterations=2)[0, 0]
        if candidate_mask.amax().item() <= 0:
            refined_binary[batch_index : batch_index + 1] = binary_mask[batch_index : batch_index + 1]
            refined_alpha[batch_index : batch_index + 1] = alpha_map[batch_index : batch_index + 1] * binary_mask[
                batch_index : batch_index + 1
            ]
            continue

        components = _connected_components_2d(candidate_mask)
        if not components:
            refined_binary[batch_index : batch_index + 1] = binary_mask[batch_index : batch_index + 1]
            refined_alpha[batch_index : batch_index + 1] = alpha_map[batch_index : batch_index + 1] * binary_mask[
                batch_index : batch_index + 1
            ]
            continue

        best_component = None
        best_score = None
        for component in components:
            core_overlap = float((component * batch_core).sum().item())
            alpha_mass = float((component * batch_alpha).sum().item())
            area = float(component.sum().item())
            center_distance_penalty = float(
                ((component > 0).float() * (1.0 - batch_core)).sum().item() / max(area, 1.0)
            )
            score = (core_overlap * 1000.0) + (alpha_mass * 10.0) - center_distance_penalty
            if best_score is None or score > best_score:
                best_score = score
                best_component = component

        assert best_component is not None
        best_component = _morphology_refine(best_component.unsqueeze(0).unsqueeze(0))[0, 0]
        refined_binary[batch_index, 0] = best_component
        refined_alpha[batch_index, 0] = batch_alpha * best_component
    return refined_alpha, refined_binary


def build_support_guided_soft_alpha(
    alpha_map: torch.Tensor,
    support_mask: torch.Tensor,
    region_mask: torch.Tensor,
    poisson_lambda: float = 0.50,
) -> torch.Tensor:
    normalized_alpha = alpha_map / alpha_map.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
    support_mask = (support_mask > 0).to(dtype=alpha_map.dtype)
    region_mask = (region_mask > 0).to(dtype=alpha_map.dtype)

    support_dilated = _binary_dilate(support_mask, iterations=1) * region_mask
    support_eroded = _binary_erode(support_mask, iterations=1) * support_mask
    boundary_band = (support_dilated - support_eroded).clamp(min=0.0) * region_mask

    smoothed_support = _screened_poisson_smooth(
        support_dilated.to(dtype=alpha_map.dtype),
        poisson_lambda=max(float(poisson_lambda), 1.0),
        num_iters=12,
    )
    smoothed_support = smoothed_support * support_dilated
    smoothed_support = smoothed_support / smoothed_support.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-6)

    # Separate support from boundary:
    # - interior support remains high-confidence writeback area
    # - the boundary band follows the raw alpha score to preserve a more natural outline
    #   instead of collapsing the whole region into a single hard blob.
    interior_alpha = support_eroded
    boundary_alpha = (0.35 * smoothed_support + 0.65 * normalized_alpha) * boundary_band
    soft_alpha = torch.maximum(interior_alpha, boundary_alpha)
    soft_alpha = soft_alpha * support_dilated
    return soft_alpha.clamp_(0.0, 1.0)


def estimate_alpha_from_token_difference(
    branch_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
    token_indices: Sequence[int] | torch.Tensor,
    token_shape: tuple[int, int, int],
    gamma: float = 0.90,
    poisson_lambda: float = 0.50,
    return_binary_mask: bool = False,
    core_first: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if branch_tokens.shape != current_tokens.shape:
        raise ValueError(
            f"branch/current token shapes must match, got {tuple(branch_tokens.shape)} vs {tuple(current_tokens.shape)}"
        )
    if branch_tokens.ndim != 3:
        raise ValueError(f"expected token tensors of shape [B, N, D], got {tuple(branch_tokens.shape)}")

    diff = torch.linalg.vector_norm(branch_tokens - current_tokens, dim=-1)
    diff_map = _scatter_token_values(diff, token_indices, token_shape)
    region_mask = _scatter_token_mask(token_indices, token_shape, branch_tokens.device)

    score_map = torch.zeros_like(diff_map)
    flat_region_mask = region_mask.view(-1).bool()
    for batch_index in range(diff_map.shape[0]):
        batch_diff_map = diff_map[batch_index : batch_index + 1]
        flat_diff = batch_diff_map.view(-1)
        region_values = flat_diff[flat_region_mask]
        if region_values.numel() == 0:
            continue

        # Coarse foreground from branch-global difference, then use its surrounding ring as local background.
        coarse_threshold = _otsu_threshold(region_values)
        coarse_fg = ((batch_diff_map >= coarse_threshold).float() * region_mask).float()
        surrounding_bg = (_binary_dilate(coarse_fg, iterations=1) - coarse_fg).clamp(min=0.0) * region_mask
        if surrounding_bg.amax().item() <= 0:
            surrounding_bg = (region_mask - coarse_fg).clamp(min=0.0)

        bg_values = flat_diff[surrounding_bg.view(-1).bool()]
        if bg_values.numel() == 0:
            bg_values = region_values

        median = bg_values.median()
        mad = (bg_values - median).abs().median()
        sigma_bg = (1.4826 * mad).clamp(min=1e-5)
        score_map[batch_index : batch_index + 1] = ((batch_diff_map / sigma_bg).pow(float(2.0 * gamma))) * region_mask

    alpha_map = _screened_poisson_smooth(score_map, poisson_lambda=poisson_lambda)
    alpha_map = alpha_map * region_mask

    alpha_tokens = []
    binary_tokens = []
    flat_alpha = alpha_map.view(alpha_map.shape[0], -1)
    for batch_index in range(alpha_map.shape[0]):
        region_values = flat_alpha[batch_index][flat_region_mask]
        threshold = _otsu_threshold(region_values)
        percentile_threshold = float(torch.quantile(region_values, 0.65).item()) if region_values.numel() > 1 else threshold
        threshold = max(threshold, percentile_threshold)
        binary = ((alpha_map[batch_index : batch_index + 1] >= threshold).float() * region_mask).float()
        binary = _morphology_refine(binary)
        tightened_binary = _binary_erode(binary, iterations=1)
        if tightened_binary.amax().item() > 0:
            binary = tightened_binary
        refined = alpha_map[batch_index : batch_index + 1]
        if core_first:
            refined, binary = refine_alpha_mask_with_region_core(
                refined,
                binary,
                token_indices,
                token_shape,
            )
        soft_alpha = build_support_guided_soft_alpha(
            refined,
            binary,
            region_mask,
            poisson_lambda=poisson_lambda,
        )
        if soft_alpha.amax().item() <= 1e-6:
            soft_alpha = refined * binary
        soft_alpha = soft_alpha / soft_alpha.amax().clamp(min=1e-6)
        alpha_tokens.append(soft_alpha.view(-1)[flat_region_mask])
        binary_tokens.append(binary.view(-1)[flat_region_mask])

    alpha = torch.stack(alpha_tokens, dim=0).to(dtype=branch_tokens.dtype, device=branch_tokens.device)
    alpha = alpha.clamp_(0.0, 1.0).unsqueeze(-1)
    if not return_binary_mask:
        return alpha

    binary_mask = torch.stack(binary_tokens, dim=0).to(dtype=branch_tokens.dtype, device=branch_tokens.device)
    binary_mask = (binary_mask > 0).to(dtype=branch_tokens.dtype).unsqueeze(-1)
    return alpha, binary_mask


def populate_region_token_indices(
    layout: LayerBindLayout,
    image_width: int,
    image_height: int,
    vae_scale_factor: int = zimage_config.ZIMAGE_VAE_SCALE_FACTOR,
    patch_size: int = zimage_config.DEFAULT_TRANSFORMER_PATCH_SIZE[0],
) -> LayerBindLayout:
    populated_regions = []
    for region in sorted(layout.regions, key=lambda item: item.layer_index):
        token_indices = sorted(
            set(
                bbox_to_token_indices(
                    region.bbox,
                    image_width=image_width,
                    image_height=image_height,
                    vae_scale_factor=vae_scale_factor,
                    patch_size=patch_size,
                )
            )
        )
        populated_regions.append(
            RegionLayer(
                region_prompt=region.region_prompt,
                bbox=region.bbox,
                layer_index=region.layer_index,
                token_indices=token_indices,
            )
        )

    return LayerBindLayout(
        background_prompt=layout.background_prompt,
        scene_prompt=layout.scene_prompt,
        negative_prompt=layout.negative_prompt,
        config=layout.config,
        regions=sorted(populated_regions, key=lambda item: item.layer_index),
    )
