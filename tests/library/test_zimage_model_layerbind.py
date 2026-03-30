import torch

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


def test_prepare_and_split_unified_tokens_round_trip():
    model = create_tiny_zimage_model()
    x = torch.randn(1, 16, 1, 4, 4)
    cap_feats = torch.randn(1, 3, 12)
    cap_mask = torch.tensor([[True, True, False]])
    t = torch.tensor([0.5])

    adaln_input = model.prepare_adaln_input(t)
    x_tokens, x_freqs_cis, meta = model.prepare_image_tokens(x, cap_seq_len=cap_feats.shape[1], adaln_input=adaln_input)
    cap_tokens, cap_freqs_cis = model.prepare_caption_tokens(cap_feats, cap_mask)
    unified, unified_freqs_cis = model.build_unified_tokens(x_tokens, x_freqs_cis, cap_tokens, cap_freqs_cis)
    x_tokens_out, cap_tokens_out = model.split_unified_tokens(unified, meta["seq_len"])

    assert x_tokens.shape == (1, 4, 32)
    assert x_freqs_cis.shape == (1, 4, 4)
    assert meta["seq_len"] == 4
    assert unified.shape == (1, 7, 32)
    assert unified_freqs_cis.shape == (1, 7, 4)
    torch.testing.assert_close(x_tokens, x_tokens_out)
    torch.testing.assert_close(cap_tokens, cap_tokens_out)


def test_select_and_replace_token_subset():
    model = create_tiny_zimage_model()
    tokens = torch.arange(1 * 4 * 32, dtype=torch.float32).view(1, 4, 32)
    freqs_cis = torch.arange(1 * 4 * 4, dtype=torch.float32).view(1, 4, 4)

    subset_tokens, subset_freqs = model.select_token_subset(tokens, [1, 3], freqs_cis)
    replacement = torch.zeros_like(subset_tokens)
    updated = model.replace_token_subset(tokens, [1, 3], replacement)

    assert subset_tokens.shape == (1, 2, 32)
    assert subset_freqs.shape == (1, 2, 4)
    torch.testing.assert_close(updated[:, 1], torch.zeros(1, 32))
    torch.testing.assert_close(updated[:, 3], torch.zeros(1, 32))
    torch.testing.assert_close(updated[:, 0], tokens[:, 0])
    torch.testing.assert_close(updated[:, 2], tokens[:, 2])


def test_helper_path_matches_forward():
    model = create_tiny_zimage_model()
    x = torch.randn(1, 16, 1, 4, 4)
    cap_feats = torch.randn(1, 3, 12)
    cap_mask = torch.tensor([[True, True, False]])
    t = torch.tensor([0.5])

    direct = model(x=x, t=t, cap_feats=cap_feats, cap_mask=cap_mask, patch_size=2, f_patch_size=1)

    adaln_input = model.prepare_adaln_input(t)
    x_tokens, x_freqs_cis, meta = model.prepare_image_tokens(
        x,
        cap_seq_len=cap_feats.shape[1],
        patch_size=2,
        f_patch_size=1,
        adaln_input=adaln_input,
    )
    adaln_input = adaln_input.type_as(x_tokens)
    cap_tokens, cap_freqs_cis = model.prepare_caption_tokens(cap_feats, cap_mask)
    unified, unified_freqs_cis = model.build_unified_tokens(x_tokens, x_freqs_cis, cap_tokens, cap_freqs_cis)
    unified = model.run_main_layers(unified, unified_freqs_cis, adaln_input, cap_mask, meta["seq_len"])
    helper = model.finalize_image_tokens(unified, adaln_input, meta["image_shape"], patch_size=2, f_patch_size=1)

    torch.testing.assert_close(direct, helper)


def test_prepare_caption_tokens_keeps_model_dtype_when_padding_is_present():
    model = create_tiny_zimage_model().to(dtype=torch.bfloat16)
    cap_feats = torch.randn(1, 3, 12, dtype=torch.bfloat16)
    cap_mask = torch.tensor([[True, True, False]])

    cap_tokens, cap_freqs_cis = model.prepare_caption_tokens(cap_feats, cap_mask, apply_context_refiner=False)

    assert cap_tokens.dtype == torch.bfloat16
    assert cap_freqs_cis.dtype == torch.bfloat16
