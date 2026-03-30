import json
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image

import zimage_minimal_inference


class FakeTransformer:
    in_channels = 16

    def __call__(self, x, t, cap_feats, cap_mask):
        return torch.zeros_like(x)


class FakeLayerBindTransformer(FakeTransformer):
    def prepare_image_tokens(self, *args, **kwargs):
        raise AssertionError("prepare_image_tokens should not be called in this scheduling test")


class FakeVAE:
    dtype = torch.float32


def test_generate_image_smoke_with_layerbind_layout(monkeypatch, tmp_path):
    encoded_prompts = []

    def fake_encode_prompt(tokenize_strategy, encoding_strategy, text_encoder, prompt, _unused, device, dtype):
        encoded_prompts.append(prompt)
        embeds = torch.ones(1, 4, 8, device=device, dtype=dtype)
        mask = torch.tensor([[True, True, True, True]], device=device)
        return embeds, mask

    layout_path = tmp_path / "layout.json"
    layout_path.write_text(
        json.dumps(
            {
                "scene_prompt": "scene from layout",
                "regions": [
                    {
                        "region_prompt": "object",
                        "bbox": [0, 0, 16, 16],
                        "layer_index": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(zimage_minimal_inference.zimage_train_utils, "_encode_prompt", fake_encode_prompt)
    monkeypatch.setattr(
        zimage_minimal_inference.zimage_train_utils,
        "_get_timesteps_sigmas",
        lambda steps, shift: (torch.tensor([1000.0]), torch.tensor([1.0, 0.0])),
    )
    monkeypatch.setattr(zimage_minimal_inference.zimage_train_utils, "_step", lambda model_output, sample, sigmas, step_index: sample)
    monkeypatch.setattr(zimage_minimal_inference.zimage_train_utils, "_unscale_latents", lambda latents, vae: latents)
    monkeypatch.setattr(zimage_minimal_inference.zimage_train_utils, "_decode_latents", lambda vae, latents: latents[:, :3])
    monkeypatch.setattr(zimage_minimal_inference.torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        zimage_minimal_inference.zimage_train_utils,
        "_latents_to_pil",
        lambda latents: Image.new("RGB", (4, 4), color="white"),
    )

    output_dir = tmp_path / "out"
    zimage_minimal_inference.generate_image(
        transformer=FakeTransformer(),
        vae=FakeVAE(),
        text_encoder=object(),
        tokenize_strategy=object(),
        encoding_strategy=object(),
        prompt_dict={
            "prompt": "base prompt",
            "negative_prompt": "",
            "guidance_scale": 1.0,
            "seed": 123,
            "sample_steps": 1,
            "width": 64,
            "height": 64,
            "layerbind_layout": str(layout_path),
        },
        output_dir=str(output_dir),
        output_name="smoke",
        steps=1,
        discrete_flow_shift=3.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    files = list(Path(output_dir).glob("*.png"))
    assert len(files) == 1
    assert encoded_prompts[0] == "scene from layout"


def test_generate_image_only_blends_phase1_at_t1(monkeypatch, tmp_path):
    def fake_encode_prompt(tokenize_strategy, encoding_strategy, text_encoder, prompt, _unused, device, dtype):
        embeds = torch.ones(1, 4, 8, device=device, dtype=dtype)
        mask = torch.tensor([[True, True, True, True]], device=device)
        return embeds, mask

    layout_path = tmp_path / "layout.json"
    layout_path.write_text(
        json.dumps(
            {
                "scene_prompt": "scene from layout",
                "config": {"eta1": 0.5, "eta2": 0.75},
                "regions": [{"region_prompt": "object", "bbox": [0, 0, 16, 16], "layer_index": 1}],
            }
        ),
        encoding="utf-8",
    )

    apply_phase1_blend_flags = []

    def fake_prepare_layerbind_conditions(*args, **kwargs):
        return {
            "scene": {"embeds": torch.ones(1, 4, 8), "mask": torch.ones(1, 4, dtype=torch.bool)},
            "background": {"embeds": torch.ones(1, 4, 8), "mask": torch.ones(1, 4, dtype=torch.bool)},
            "regions": [{"tokens": torch.ones(1, 4, 8), "freqs": torch.ones(1, 4, 3)}],
            "negative": None,
            "image_sequence_length": 16,
        }

    def fake_run_layerbind_forward(
        transformer,
        latent_model_input,
        timestep,
        scene_condition,
        background_condition,
        region_conditions,
        region_states,
        phase,
        hard_binding_layers,
        beta,
        blend_mode,
        gamma,
        poisson_lambda,
        apply_phase1_blend,
    ):
        if phase == "phase1":
            apply_phase1_blend_flags.append(apply_phase1_blend)
        return torch.zeros_like(latent_model_input)

    monkeypatch.setattr(zimage_minimal_inference.zimage_train_utils, "_encode_prompt", fake_encode_prompt)
    monkeypatch.setattr(
        zimage_minimal_inference.zimage_train_utils,
        "_get_timesteps_sigmas",
        lambda steps, shift: (torch.tensor([1000.0, 700.0, 400.0, 100.0]), torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])),
    )
    monkeypatch.setattr(zimage_minimal_inference.zimage_train_utils, "_step", lambda model_output, sample, sigmas, step_index: sample)
    monkeypatch.setattr(zimage_minimal_inference.zimage_train_utils, "_unscale_latents", lambda latents, vae: latents)
    monkeypatch.setattr(zimage_minimal_inference.zimage_train_utils, "_decode_latents", lambda vae, latents: latents[:, :3])
    monkeypatch.setattr(zimage_minimal_inference.zimage_train_utils, "_latents_to_pil", lambda latents: Image.new("RGB", (4, 4)))
    monkeypatch.setattr(zimage_minimal_inference.torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(zimage_minimal_inference, "prepare_layerbind_conditions", fake_prepare_layerbind_conditions)
    monkeypatch.setattr(
        zimage_minimal_inference,
        "prepare_region_runtime_states",
        lambda layout, x_seq_len, device: [{"indices": torch.tensor([0]), "background_indices": torch.tensor([1])}],
    )
    monkeypatch.setattr(zimage_minimal_inference, "run_layerbind_forward", fake_run_layerbind_forward)

    output_dir = tmp_path / "out_schedule"
    zimage_minimal_inference.generate_image(
        transformer=FakeLayerBindTransformer(),
        vae=FakeVAE(),
        text_encoder=object(),
        tokenize_strategy=object(),
        encoding_strategy=object(),
        prompt_dict={
            "prompt": "base prompt",
            "negative_prompt": "",
            "guidance_scale": 1.0,
            "seed": 123,
            "sample_steps": 4,
            "width": 64,
            "height": 64,
            "layerbind_layout": str(layout_path),
        },
        output_dir=str(output_dir),
        output_name="schedule",
        steps=4,
        discrete_flow_shift=3.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert apply_phase1_blend_flags == [False, True]
