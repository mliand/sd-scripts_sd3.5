import torch

from library.zimage_layerbind_utils import (
    LayerBindLayout,
    RegionLayer,
    bbox_to_token_bounds,
    bbox_to_token_indices,
    estimate_alpha_from_token_difference,
    get_token_grid_size,
    populate_region_token_indices,
    token_indices_to_mask,
)


def test_get_token_grid_size_for_1024_square_image():
    assert get_token_grid_size(1024, 1024) == (64, 64)


def test_bbox_to_token_bounds_maps_one_visual_patch():
    assert bbox_to_token_bounds((0, 0, 16, 16), 1024, 1024) == (0, 0, 1, 1)


def test_bbox_to_token_indices_clamps_and_sorts_coordinates():
    indices = bbox_to_token_indices((32, 16, 0, 0), 1024, 1024)
    assert indices == [0, 1]


def test_token_indices_to_mask_uses_expected_grid_shape():
    mask = token_indices_to_mask([0, 3], 1024, 1024)
    assert mask.shape == (64, 64)
    assert mask[0, 0].item() is True
    assert mask[0, 3].item() is True
    assert mask.sum().item() == 2


def test_populate_region_token_indices_preserves_layer_order():
    layout = LayerBindLayout(
        scene_prompt="scene",
        regions=[
            RegionLayer(region_prompt="front", bbox=(16, 0, 32, 16), layer_index=2),
            RegionLayer(region_prompt="back", bbox=(0, 0, 16, 16), layer_index=1),
        ],
    )

    populated = populate_region_token_indices(layout, image_width=1024, image_height=1024)

    assert [region.layer_index for region in populated.regions] == [1, 2]
    assert populated.regions[0].token_indices == [0]
    assert populated.regions[1].token_indices == [1]


def test_populate_region_token_indices_preserves_overlap_for_compositing():
    layout = LayerBindLayout(
        scene_prompt="scene",
        regions=[
            RegionLayer(region_prompt="back", bbox=(0, 0, 32, 16), layer_index=1),
            RegionLayer(region_prompt="front", bbox=(16, 0, 48, 16), layer_index=2),
        ],
    )

    populated = populate_region_token_indices(layout, image_width=1024, image_height=1024)

    assert [region.layer_index for region in populated.regions] == [1, 2]
    assert populated.regions[0].token_indices == [0, 1]
    assert populated.regions[1].token_indices == [1, 2]


def test_estimate_alpha_from_token_difference_highlights_changed_tokens():
    current = torch.zeros(1, 2, 4)
    branch = current.clone()
    branch[:, 0] = 4.0

    alpha = estimate_alpha_from_token_difference(
        branch,
        current,
        token_indices=[0, 1],
        token_shape=(1, 2, 2),
        gamma=0.9,
        poisson_lambda=0.5,
        beta=1.0,
    )

    assert alpha.shape == (1, 2, 1)
    assert alpha[0, 0, 0].item() > alpha[0, 1, 0].item()
    assert 0.0 <= alpha[0, 0, 0].item() <= 1.0
