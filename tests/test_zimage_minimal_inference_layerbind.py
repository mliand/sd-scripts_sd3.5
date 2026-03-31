import json
import types
from argparse import Namespace

import torch

import zimage_minimal_inference
from library.zimage_model import ZImageTransformer2DModel


def create_tiny_zimage_model():
    model = ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=32,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=4,
        n_kv_heads=4,
        norm_eps=1e-5,
        qk_norm=False,
        cap_feat_dim=12,
        axes_dims=[2, 2, 4],
        axes_lens=[32, 8, 8],
        attn_mode="torch",
        split_attn=False,
        gate_type="none",
    )
    model.eval()
    return model


def test_parse_layer_spec_supports_ranges_and_single_values():
    assert zimage_minimal_inference.parse_layer_spec("0,2 4-6") == [0, 2, 4, 5, 6]


def test_normalize_layer_indices_keeps_none_and_deduplicates():
    assert zimage_minimal_inference.normalize_layer_indices(None) is None
    assert zimage_minimal_inference.normalize_layer_indices([3, 1, 3]) == [1, 3]


def test_prepare_layerbind_layout_populates_token_indices_and_overrides_config(tmp_path):
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(
        json.dumps(
            {
                "background_prompt": "background",
                "scene_prompt": "scene override",
                "negative_prompt": "bad quality",
                "regions": [
                    {
                        "region_prompt": "front object",
                        "bbox": [0, 0, 16, 16],
                        "layer_index": 2,
                    },
                    {
                        "region_prompt": "back object",
                        "bbox": [16, 0, 32, 16],
                        "layer_index": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    layout = zimage_minimal_inference.prepare_layerbind_layout(
        {
            "layerbind_layout": str(layout_path),
            "layerbind_eta1": 0.25,
            "layerbind_eta2": 0.75,
            "layerbind_beta": 0.55,
        },
        width=1024,
        height=1024,
    )

    assert layout.scene_prompt == "scene override"
    assert layout.negative_prompt == "bad quality"
    assert [region.layer_index for region in layout.regions] == [1, 2]
    assert layout.regions[0].token_indices == [1]
    assert layout.regions[1].token_indices == [0]
    assert layout.config.eta1 == 0.25
    assert layout.config.eta2 == 0.75
    assert layout.config.beta == 0.55


def test_default_hard_binding_layers_follow_zimage_layer_search():
    assert zimage_minimal_inference.get_default_layerbind_hard_binding_layers(30) == [0, 15, 16, 18, 19, 20, 27, 28, 29]


def test_build_prompts_includes_layerbind_fields():
    args = Namespace(
        sample_prompts=None,
        prompt="prompt",
        negative_prompt="neg",
        guidance_scale=4.0,
        seed=7,
        steps=8,
        width=1024,
        height=1024,
        layerbind_layout="layout.json",
        layerbind_eta1=0.2,
        layerbind_eta2=0.7,
        layerbind_beta=0.7,
        layerbind_hard_binding_layers=None,
        layerbind_blend_mode="alpha",
        layerbind_save_intermediates=False,
        layerbind_collect_layer_stats=True,
        layerbind_layer_stats_path="layer_stats.json",
        layerbind_layer_stats_top_k=7,
    )

    prompts = zimage_minimal_inference.build_prompts(args)

    assert len(prompts) == 1
    assert prompts[0]["layerbind_layout"] == "layout.json"
    assert prompts[0]["layerbind_eta1"] == 0.2
    assert prompts[0]["layerbind_collect_layer_stats"] is True
    assert prompts[0]["layerbind_layer_stats_path"] == "layer_stats.json"
    assert prompts[0]["layerbind_layer_stats_top_k"] == 7


def test_create_image_freqs_for_caption_length_matches_requested_offset():
    class FakeTransformer:
        def create_image_position_ids(self, f_tokens, h_tokens, w_tokens, cap_seq_len, device):
            return zimage_minimal_inference.torch.tensor(
                [[cap_seq_len + 1, 0, 0], [cap_seq_len + 1, 0, 1]],
                device=device,
                dtype=zimage_minimal_inference.torch.int32,
            )

        def rope_embedder(self, position_ids):
            return position_ids.to(dtype=zimage_minimal_inference.torch.float32)

    freqs = zimage_minimal_inference.create_image_freqs_for_caption_length(
        FakeTransformer(),
        token_shape=(1, 1, 2),
        cap_seq_len=5,
        batch_size=1,
        device=zimage_minimal_inference.torch.device("cpu"),
    )

    assert freqs.shape == (1, 2, 3)
    assert freqs[0, 0, 0].item() == 6.0


def test_create_region_local_freqs_rebases_to_region_origin():
    class FakeTransformer:
        def rope_embedder(self, position_ids):
            return position_ids.to(dtype=torch.float32)

    freqs = zimage_minimal_inference.create_region_local_freqs_for_caption_length(
        FakeTransformer(),
        token_shape=(1, 4, 4),
        token_indices=torch.tensor([5, 6, 9, 10], dtype=torch.long),
        cap_seq_len=7,
        batch_size=1,
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [[[8.0, 0.0, 0.0], [8.0, 0.0, 1.0], [8.0, 1.0, 0.0], [8.0, 1.0, 1.0]]]
    )
    assert torch.allclose(freqs, expected)


def test_build_layerbind_local_context_indices_returns_full_global_context():
    indices = zimage_minimal_inference.build_layerbind_local_context_indices(
        region_indices=torch.tensor([1, 3], dtype=torch.long),
        token_shape=(1, 2, 3),
        seq_len=6,
        device=torch.device("cpu"),
        forbidden_indices=torch.tensor([0, 5], dtype=torch.long),
        radius=1,
        global_anchor_count=2,
    )

    assert torch.equal(indices, torch.tensor([0, 2, 4, 5], dtype=torch.long))


def test_prepare_layerbind_layout_keeps_overlap_tokens_for_all_layers(tmp_path):
    layout_path = tmp_path / "layout_overlap.json"
    layout_path.write_text(
        json.dumps(
            {
                "regions": [
                    {"region_prompt": "back", "bbox": [0, 0, 32, 16], "layer_index": 1},
                    {"region_prompt": "front", "bbox": [16, 0, 48, 16], "layer_index": 2},
                ]
            }
        ),
        encoding="utf-8",
    )

    layout = zimage_minimal_inference.prepare_layerbind_layout({"layerbind_layout": str(layout_path)}, width=48, height=16)

    assert layout.regions[0].token_indices == [0, 1]
    assert layout.regions[1].token_indices == [1, 2]


def test_phase1_branch_state_evolves_independently_from_current_global_patches():
    transformer = create_tiny_zimage_model()
    device = torch.device("cpu")
    cap_mask = torch.tensor([[True, True, True]], device=device)
    cap_feats = torch.randn(1, 3, 12, device=device)
    scene_tokens, scene_freqs = transformer.prepare_caption_tokens(cap_feats, cap_mask, apply_context_refiner=False)
    region_tokens, region_freqs = transformer.prepare_caption_tokens(cap_feats, cap_mask, apply_context_refiner=False)

    scene_condition = {"tokens": scene_tokens.clone(), "mask": cap_mask, "freqs": scene_freqs}
    background_condition = {"tokens": scene_tokens.clone(), "mask": cap_mask, "freqs": scene_freqs}
    region_conditions = [{"tokens": region_tokens.clone(), "freqs": region_freqs}]
    region_states = [
        {
            "layer_index": 1,
            "bbox": (0, 0, 16, 16),
            "prompt": "object",
            "indices": torch.tensor([0], device=device),
            "background_indices": torch.tensor([1, 2, 3], device=device),
            "branch_patches": None,
            "branch_tokens": None,
            "text_tokens": None,
            "alpha_mask": None,
        }
    ]

    sigmas = torch.tensor([1.0, 0.5, 0.0], device=device)
    latent_a = torch.randn(1, 16, 1, 4, 4, device=device)
    latent_b = torch.randn(1, 16, 1, 4, 4, device=device)

    zimage_minimal_inference.run_layerbind_forward(
        transformer,
        latent_a,
        torch.tensor([0.5], device=device),
        sigmas,
        0,
        scene_condition,
        background_condition,
        region_conditions,
        region_states,
        phase="phase1",
        hard_binding_layers=[0],
        beta=0.7,
        blend_mode="alpha",
        gamma=0.9,
        poisson_lambda=0.5,
        phase2_delta_scale=0.5,
        apply_phase1_blend=False,
    )
    first_branch_patches = region_states[0]["branch_patches"].clone()

    zimage_minimal_inference.run_layerbind_forward(
        transformer,
        latent_b,
        torch.tensor([0.25], device=device),
        sigmas,
        1,
        scene_condition,
        background_condition,
        region_conditions,
        region_states,
        phase="phase1",
        hard_binding_layers=[0],
        beta=0.7,
        blend_mode="alpha",
        gamma=0.9,
        poisson_lambda=0.5,
        phase2_delta_scale=0.5,
        apply_phase1_blend=False,
    )

    second_branch_patches = region_states[0]["branch_patches"]
    current_global_region = transformer.patchify(latent_b, 2, 1).index_select(1, region_states[0]["indices"])

    assert second_branch_patches.shape == current_global_region.shape
    assert not torch.allclose(first_branch_patches, second_branch_patches)
    assert not torch.allclose(second_branch_patches, current_global_region)


def test_finalize_layerbind_layer_stats_suggests_text_dominant_layers():
    accumulator = {
        "num_layers": 4,
        "top_k": 3,
        "layout_scene_prompt": "scene",
        "layout_background_prompt": "background",
        "regions": [],
        "layers": {
            "0": {
                "self_attention_sum": 0.5,
                "background_attention_sum": 1.0,
                "text_attention_sum": 2.0,
                "query_vector_count": 2.0,
            },
            "1": {
                "self_attention_sum": 0.5,
                "background_attention_sum": 3.0,
                "text_attention_sum": 1.0,
                "query_vector_count": 2.0,
            },
            "2": {
                "self_attention_sum": 0.5,
                "background_attention_sum": 1.0,
                "text_attention_sum": 5.0,
                "query_vector_count": 2.0,
            },
            "3": {
                "self_attention_sum": 0.5,
                "background_attention_sum": 1.0,
                "text_attention_sum": 4.0,
                "query_vector_count": 2.0,
            },
        },
    }

    summary = zimage_minimal_inference.finalize_layerbind_layer_stats(accumulator)

    assert summary["suggested_hard_binding_layers"] == [0, 2, 3]


def test_get_layerbind_debug_save_points_uses_requested_percents():
    assert zimage_minimal_inference.get_layerbind_debug_save_points(30) == [(3, 10), (9, 30), (18, 60), (24, 80)]
    assert zimage_minimal_inference.get_layerbind_debug_save_points(4) == [(1, 10), (2, 30), (3, 60), (4, 80)]


def test_phase1_blend_keeps_bottom_layer_direct_in_alpha_mode(monkeypatch):
    x_tokens = torch.zeros(1, 1, 1)
    region_states = [
        {
            "layer_index": 1,
            "indices": torch.tensor([0], dtype=torch.long),
            "branch_tokens": torch.tensor([[[4.0]]]),
            "alpha_mask": None,
        },
        {
            "layer_index": 2,
            "indices": torch.tensor([0], dtype=torch.long),
            "branch_tokens": torch.tensor([[[2.0]]]),
            "alpha_mask": None,
        },
    ]

    monkeypatch.setattr(
        zimage_minimal_inference.zimage_layerbind_utils,
        "estimate_alpha_from_token_difference",
        lambda *args, **kwargs: (torch.tensor([[[0.25]]]), torch.tensor([[[1.0]]])),
    )

    blended = zimage_minimal_inference.blend_region_tokens(
        x_tokens,
        region_states,
        beta=0.7,
        blend_mode="alpha",
        token_shape=(1, 1, 1),
        gamma=0.9,
        poisson_lambda=0.5,
    )

    assert blended[0, 0, 0].item() == 3.5
    assert region_states[0]["alpha_mask"] is None
    assert abs(region_states[0]["region_mask"][0, 0, 0].item() - 1.0) < 1e-6
    assert abs(region_states[1]["alpha_mask"][0, 0, 0].item() - 0.25) < 1e-6
    assert abs(region_states[1]["region_mask"][0, 0, 0].item() - 1.0) < 1e-6
    assert region_states[0]["is_occluding"] is False
    assert region_states[1]["is_occluding"] is True


def test_phase1_blend_uses_direct_for_non_overlapping_layers(monkeypatch):
    x_tokens = torch.zeros(1, 2, 1)
    region_states = [
        {
            "layer_index": 1,
            "indices": torch.tensor([0], dtype=torch.long),
            "branch_tokens": torch.tensor([[[4.0]]]),
            "alpha_mask": None,
        },
        {
            "layer_index": 2,
            "indices": torch.tensor([1], dtype=torch.long),
            "branch_tokens": torch.tensor([[[2.0]]]),
            "alpha_mask": None,
        },
    ]

    monkeypatch.setattr(
        zimage_minimal_inference.zimage_layerbind_utils,
        "estimate_alpha_from_token_difference",
        lambda *args, **kwargs: (torch.tensor([[[0.25]]]), torch.tensor([[[1.0]]])),
    )

    blended = zimage_minimal_inference.blend_region_tokens(
        x_tokens,
        region_states,
        beta=0.7,
        blend_mode="alpha",
        token_shape=(1, 1, 2),
        gamma=0.9,
        poisson_lambda=0.5,
    )

    assert blended[0, 0, 0].item() == 4.0
    assert blended[0, 1, 0].item() == 2.0
    assert region_states[1]["alpha_mask"] is None
    assert region_states[1]["is_occluding"] is False
    assert abs(region_states[1]["region_mask"][0, 0, 0].item() - 1.0) < 1e-6


def test_phase1_blend_preserves_global_background_for_bottom_layers_in_alpha_mode(monkeypatch):
    x_tokens = torch.tensor([[[5.0]]])
    region_states = [
        {
            "layer_index": 1,
            "indices": torch.tensor([0], dtype=torch.long),
            "branch_tokens": torch.tensor([[[9.0]]]),
            "alpha_mask": None,
        }
    ]

    monkeypatch.setattr(
        zimage_minimal_inference.zimage_layerbind_utils,
        "estimate_alpha_from_token_difference",
        lambda *args, **kwargs: (torch.tensor([[[0.25]]]), torch.tensor([[[0.0]]])),
    )

    blended = zimage_minimal_inference.blend_region_tokens(
        x_tokens,
        region_states,
        beta=0.7,
        blend_mode="alpha",
        token_shape=(1, 1, 1),
        gamma=0.9,
        poisson_lambda=0.5,
    )

    assert abs(blended[0, 0, 0].item() - 5.0) < 1e-6
    assert region_states[0]["alpha_mask"] is None
    assert abs(region_states[0]["region_mask"][0, 0, 0].item() - 0.0) < 1e-6


def test_phase2_composition_uses_beta_times_binary_mask_once():
    x_tokens = torch.zeros(1, 2, 1)
    local_tokens = torch.tensor([[[10.0], [10.0]]])
    indices = torch.tensor([0, 1], dtype=torch.long)
    region_mask = torch.tensor([[[1.0], [0.0]]])

    composed = zimage_minimal_inference.compose_phase2_region_tokens(
        x_tokens,
        local_tokens,
        indices,
        beta=0.7,
        region_mask=region_mask,
    )

    assert abs(composed[0, 0, 0].item() - 7.0) < 1e-6
    assert abs(composed[0, 1, 0].item() - 0.0) < 1e-6


def test_phase1_resets_region_text_tokens_from_prompt_each_timestep(monkeypatch):
    transformer = create_tiny_zimage_model()
    device = torch.device("cpu")
    cap_mask = torch.tensor([[True, True, True]], device=device)
    cap_feats = torch.randn(1, 3, 12, device=device)
    scene_tokens, scene_freqs = transformer.prepare_caption_tokens(cap_feats, cap_mask, apply_context_refiner=False)
    region_tokens, region_freqs = transformer.prepare_caption_tokens(cap_feats, cap_mask, apply_context_refiner=False)

    scene_condition = {"tokens": scene_tokens.clone(), "mask": cap_mask, "freqs": scene_freqs}
    background_condition = {"tokens": scene_tokens.clone(), "mask": cap_mask, "freqs": scene_freqs}
    region_conditions = [{"tokens": region_tokens.clone(), "freqs": region_freqs}]
    region_states = [
        {
            "layer_index": 1,
            "bbox": (0, 0, 16, 16),
            "prompt": "object",
            "indices": torch.tensor([0], device=device),
            "background_indices": torch.tensor([1, 2, 3], device=device),
            "foreign_region_indices": torch.zeros((0,), dtype=torch.long, device=device),
            "is_occluding_hint": False,
            "branch_patches": None,
            "branch_tokens": None,
            "text_tokens": torch.full_like(region_tokens, 9.0),
            "region_mask": torch.ones((1, 1, 1), device=device),
            "alpha_mask": None,
        }
    ]

    def fake_contextual_forward(
        self,
        query_states,
        query_freqs_cis,
        context_states=None,
        context_freqs_cis=None,
        adaln_input=None,
        attn_params=None,
        include_query_in_kv=True,
        segment_logit_biases=None,
    ):
        return query_states

    for layer in transformer.layers:
        monkeypatch.setattr(layer, "contextual_forward", types.MethodType(fake_contextual_forward, layer))

    sigmas = torch.tensor([1.0, 0.5, 0.0], device=device)
    latent = torch.randn(1, 16, 1, 4, 4, device=device)
    zimage_minimal_inference.run_layerbind_forward(
        transformer,
        latent,
        torch.tensor([0.5], device=device),
        sigmas,
        0,
        scene_condition,
        background_condition,
        region_conditions,
        region_states,
        phase="phase1",
        hard_binding_layers=[],
        beta=0.7,
        blend_mode="alpha",
        gamma=0.9,
        poisson_lambda=0.5,
        phase2_delta_scale=0.5,
        apply_phase1_blend=False,
    )

    assert torch.allclose(region_states[0]["text_tokens"], region_tokens)


def test_phase2_resets_region_text_tokens_from_prompt_each_timestep(monkeypatch):
    transformer = create_tiny_zimage_model()
    device = torch.device("cpu")
    cap_mask = torch.tensor([[True, True, True]], device=device)
    cap_feats = torch.randn(1, 3, 12, device=device)
    scene_tokens, scene_freqs = transformer.prepare_caption_tokens(cap_feats, cap_mask, apply_context_refiner=False)
    region_tokens, region_freqs = transformer.prepare_caption_tokens(cap_feats, cap_mask, apply_context_refiner=False)

    scene_condition = {"tokens": scene_tokens.clone(), "mask": cap_mask, "freqs": scene_freqs}
    background_condition = {"tokens": scene_tokens.clone(), "mask": cap_mask, "freqs": scene_freqs}
    region_conditions = [{"tokens": region_tokens.clone(), "freqs": region_freqs}]
    region_states = [
        {
            "layer_index": 1,
            "bbox": (0, 0, 16, 16),
            "prompt": "object",
            "indices": torch.tensor([0], device=device),
            "background_indices": torch.tensor([1, 2, 3], device=device),
            "foreign_region_indices": torch.zeros((0,), dtype=torch.long, device=device),
            "is_occluding_hint": False,
            "branch_patches": None,
            "branch_tokens": None,
            "text_tokens": torch.full_like(region_tokens, 9.0),
            "region_mask": torch.ones((1, 1, 1), device=device),
            "alpha_mask": None,
        }
    ]

    def fake_contextual_forward(
        self,
        query_states,
        query_freqs_cis,
        context_states=None,
        context_freqs_cis=None,
        adaln_input=None,
        attn_params=None,
        include_query_in_kv=True,
        segment_logit_biases=None,
    ):
        return query_states + 1.0

    for layer in transformer.layers:
        monkeypatch.setattr(layer, "contextual_forward", types.MethodType(fake_contextual_forward, layer))

    sigmas = torch.tensor([1.0, 0.5, 0.0], device=device)
    latent = torch.randn(1, 16, 1, 4, 4, device=device)
    zimage_minimal_inference.run_layerbind_forward(
        transformer,
        latent,
        torch.tensor([0.25], device=device),
        sigmas,
        1,
        scene_condition,
        background_condition,
        region_conditions,
        region_states,
        phase="phase2",
        hard_binding_layers=[],
        beta=0.7,
        blend_mode="alpha",
        gamma=0.9,
        poisson_lambda=0.5,
        phase2_delta_scale=0.5,
        apply_phase1_blend=False,
    )

    expected = region_tokens + zimage_minimal_inference.LAYERBIND_PHASE2_TEXT_UPDATE_SCALE * len(transformer.layers)
    assert torch.allclose(region_states[0]["text_tokens"], expected)
