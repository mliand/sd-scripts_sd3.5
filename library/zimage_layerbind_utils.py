import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from library import zimage_config


@dataclass
class LayerBindConfig:
    eta1: float = 0.20
    eta2: float = 0.70
    beta: float = 0.70
    gamma: float = 0.90
    poisson_lambda: float = 0.50
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
    for key in ("eta1", "eta2", "beta", "gamma", "poisson_lambda"):
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


def populate_region_token_indices(
    layout: LayerBindLayout,
    image_width: int,
    image_height: int,
    vae_scale_factor: int = zimage_config.ZIMAGE_VAE_SCALE_FACTOR,
    patch_size: int = zimage_config.DEFAULT_TRANSFORMER_PATCH_SIZE[0],
) -> LayerBindLayout:
    sorted_regions = sorted(layout.regions, key=lambda item: item.layer_index)
    occupied_indices: set[int] = set()
    populated_regions = []

    # Front-most regions keep overlapping tokens; back regions lose the overlap.
    for region in reversed(sorted_regions):
        token_indices = bbox_to_token_indices(
            region.bbox,
            image_width=image_width,
            image_height=image_height,
            vae_scale_factor=vae_scale_factor,
            patch_size=patch_size,
        )
        token_indices = [index for index in token_indices if index not in occupied_indices]
        occupied_indices.update(token_indices)
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
        config=layout.config,
        regions=sorted(populated_regions, key=lambda item: item.layer_index),
    )
