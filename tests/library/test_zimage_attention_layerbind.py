import torch

from library.zimage_attention import contextual_attention, attention
from library.zimage_model import ZImageAttention


def test_contextual_attention_matches_explicit_kv_concat():
    query = torch.randn(1, 2, 2, 4)
    context = torch.randn(1, 3, 2, 4)

    actual = contextual_attention(query, context)
    expected = attention(query, torch.cat([query, context], dim=1), torch.cat([query, context], dim=1))

    torch.testing.assert_close(actual, expected)


def test_contextual_attention_supports_query_excluded_from_kv():
    query = torch.randn(1, 2, 2, 4)
    context = torch.randn(1, 3, 2, 4)

    actual = contextual_attention(query, context, include_query_in_kv=False)
    expected = attention(query, context, context)

    torch.testing.assert_close(actual, expected)


def test_zimage_attention_contextual_forward_matches_forward_without_extra_context():
    module = ZImageAttention(dim=16, n_heads=4, n_kv_heads=4, qk_norm=False, gate_type="none")
    hidden_states = torch.randn(1, 3, 16)

    actual = module(hidden_states)
    expected = module.contextual_forward(hidden_states, context_states=None)

    torch.testing.assert_close(actual, expected)
