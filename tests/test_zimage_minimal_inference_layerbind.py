import json
from argparse import Namespace

import zimage_minimal_inference


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
    assert [region.layer_index for region in layout.regions] == [1, 2]
    assert layout.regions[0].token_indices == [1]
    assert layout.regions[1].token_indices == [0]
    assert layout.config.eta1 == 0.25
    assert layout.config.eta2 == 0.75
    assert layout.config.beta == 0.55


def test_default_hard_binding_layers_follow_sd35_style_density():
    assert zimage_minimal_inference.get_default_layerbind_hard_binding_layers(30) == [0, 9, 12, 16, 18, 20, 25, 27, 29]


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
    )

    prompts = zimage_minimal_inference.build_prompts(args)

    assert len(prompts) == 1
    assert prompts[0]["layerbind_layout"] == "layout.json"
    assert prompts[0]["layerbind_eta1"] == 0.2


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
