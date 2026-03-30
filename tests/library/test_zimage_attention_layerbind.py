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


def test_zimage_attention_contextual_attention_stats_track_segment_mass():
    module = ZImageAttention(dim=16, n_heads=4, n_kv_heads=4, qk_norm=False, gate_type="none")
    with torch.no_grad():
        module.to_q.weight.zero_()
        module.to_k.weight.zero_()

    query_states = torch.randn(1, 2, 16)
    background_states = torch.randn(1, 3, 16)
    text_states = torch.randn(1, 1, 16)

    stats = module.contextual_attention_stats(
        query_states,
        context_states=[background_states, text_states],
        include_query_in_kv=True,
        segment_names=["self", "background", "text"],
    )

    assert stats["query_vector_count"] == 8.0
    assert abs(stats["segment_attention/self"] - (2.0 / 6.0)) < 1e-5
    assert abs(stats["segment_attention/background"] - (3.0 / 6.0)) < 1e-5
    assert abs(stats["segment_attention/text"] - (1.0 / 6.0)) < 1e-5
