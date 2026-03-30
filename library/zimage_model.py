# Copyright (c) 2025 Z-Image Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from the original version.
# Original implementation: https://github.com/Tongyi-MAI/Z-Image
# Modifications: Copied and modified for Musubi Tuner project.

"""Z-Image Transformer."""

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from accelerate import init_empty_weights

from library.zimage_attention import AttentionParams, attention
from library import zimage_config
from library.zimage_config import (
    ADALN_EMBED_DIM,
    FREQUENCY_EMBEDDING_SIZE,
    MAX_PERIOD,
    ROPE_AXES_DIMS,
    ROPE_AXES_LENS,
    ROPE_THETA,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_cpu_offloading_wrapper(forward_fn, device):
    return forward_fn


class ModelOffloader:
    def __init__(self, *args, **kwargs) -> None:
        self.forward_only = False

    def prepare_block_devices_before_forward(self, _layers):
        return None

    def set_forward_only(self, forward_only: bool):
        self.forward_only = forward_only


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=FREQUENCY_EMBEDDING_SIZE):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=MAX_PERIOD):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # original implementation. kept for reference
        # output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # return output * self.weight

        # cast to float32 for numerical stability
        x_f = x.float()
        w_f = self.weight.float()
        out = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * w_f).to(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def forward(self, x):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


class ZImageAttention(nn.Module):
    _attention_backend = None

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-5,
        use_16bit: bool = False,
        gate_type: str = "none",
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.use_16bit = use_16bit
        self.gate_type = gate_type
        self.gate_enabled = True

        if gate_type == "headwise":
            self.gate_dim = n_heads
        elif gate_type == "elementwise":
            self.gate_dim = dim
        elif gate_type == "none":
            self.gate_dim = 0
        else:
            raise ValueError(f"Unsupported gate_type: {gate_type}")

        self.to_q = nn.Linear(dim, n_heads * self.head_dim + self.gate_dim, bias=False)
        self.to_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(n_heads * self.head_dim, dim, bias=False)])

        if self.gate_dim > 0:
            with torch.no_grad():
                self.to_q.weight[n_heads * self.head_dim :, :].zero_()

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else None

        self.gradient_checkpointing = False
        self._log_gate_stats = False
        self._last_gate_stats = None
        self._gate_grad_hook = None
        self._frozen_gate_weight = None

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def set_log_gate_stats(self, enabled: bool):
        self._log_gate_stats = enabled
        if not enabled:
            self._last_gate_stats = None

    def set_gate_enabled(self, enabled: bool):
        self.gate_enabled = enabled

    def set_gate_trainable(self, enabled: bool):
        if self.gate_dim == 0:
            return
        gate_rows = slice(self.n_heads * self.head_dim, self.n_heads * self.head_dim + self.gate_dim)
        if enabled:
            if self._gate_grad_hook is not None:
                self._gate_grad_hook.remove()
                self._gate_grad_hook = None
            self._frozen_gate_weight = None
            return

        if self._gate_grad_hook is None:
            def _gate_grad_hook(grad):
                if grad is None:
                    return grad
                grad = grad.clone()
                grad[gate_rows, :] = 0
                return grad

            self._gate_grad_hook = self.to_q.weight.register_hook(_gate_grad_hook)
        self._frozen_gate_weight = self.to_q.weight.data[gate_rows].detach().clone()

    def restore_frozen_gate(self):
        if self.gate_dim == 0 or self._frozen_gate_weight is None:
            return
        gate_rows = slice(self.n_heads * self.head_dim, self.n_heads * self.head_dim + self.gate_dim)
        with torch.no_grad():
            self.to_q.weight.data[gate_rows].copy_(self._frozen_gate_weight)

    def get_gate_statistics(self) -> Dict[str, float]:
        return self._last_gate_stats or {}

    @staticmethod
    def _compute_gate_statistics(gate_values: torch.Tensor) -> Dict[str, float]:
        stats = {
            "gate_mean": gate_values.mean().item(),
            "gate_std": gate_values.std().item(),
            "gate_min": gate_values.min().item(),
            "gate_max": gate_values.max().item(),
            "gate_sparsity": (gate_values < 0.5).float().mean().item(),
        }
        return stats

    def _split_query_and_gate(self, query: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        gate_score = None
        if self.gate_type == "headwise":
            query = query.reshape(query.shape[0], query.shape[1], self.n_heads, self.head_dim + 1)
            query, gate_score = torch.split(query, [self.head_dim, 1], dim=-1)
        elif self.gate_type == "elementwise":
            query = query.reshape(query.shape[0], query.shape[1], self.n_heads, self.head_dim * 2)
            query, gate_score = torch.split(query, [self.head_dim, self.head_dim], dim=-1)
        else:
            query = query.unflatten(-1, (self.n_heads, -1))

        if gate_score is not None and not self.gate_enabled:
            gate_score = None

        return query, gate_score

    def _project_query(
        self, hidden_states: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.dtype]:
        query = self.to_q(hidden_states)
        query, gate_score = self._split_query_and_gate(query)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)

        target_dtype = query.dtype if not self.use_16bit else hidden_states.dtype
        query = query.to(target_dtype)
        return query, gate_score, target_dtype

    def _project_key_value(
        self, hidden_states: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, dtype: Optional[torch.dtype] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.to_k(hidden_states).unflatten(-1, (self.n_kv_heads, -1))
        value = self.to_v(hidden_states).unflatten(-1, (self.n_kv_heads, -1))

        if self.norm_k is not None:
            key = self.norm_k(key)
        if freqs_cis is not None:
            key = apply_rotary_emb(key, freqs_cis)

        if dtype is None:
            dtype = key.dtype if not self.use_16bit else value.dtype
        key = key.to(dtype)
        return key, value

    def _finalize_attention_output(
        self, hidden_states: torch.Tensor, gate_score: Optional[torch.Tensor], dtype: torch.dtype
    ) -> torch.Tensor:
        if gate_score is not None:
            gate_values = torch.sigmoid(gate_score)
            hidden_states = hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], self.n_heads, self.head_dim)
            hidden_states = hidden_states * gate_values
            if self._log_gate_stats:
                with torch.no_grad():
                    self._last_gate_stats = self._compute_gate_statistics(gate_values)
            hidden_states = hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)

        hidden_states = hidden_states.to(dtype)
        return self.to_out[0](hidden_states)

    @staticmethod
    def _normalize_context_states(
        context_states: Optional[Sequence[torch.Tensor] | torch.Tensor],
    ) -> list[torch.Tensor]:
        if context_states is None:
            return []
        if isinstance(context_states, torch.Tensor):
            return [context_states]
        return list(context_states)

    @staticmethod
    def _normalize_context_freqs(
        context_freqs_cis: Optional[Sequence[torch.Tensor] | torch.Tensor],
        expected_length: int,
    ) -> list[Optional[torch.Tensor]]:
        if context_freqs_cis is None:
            return [None] * expected_length
        if isinstance(context_freqs_cis, torch.Tensor):
            freqs = [context_freqs_cis]
        else:
            freqs = list(context_freqs_cis)
        if len(freqs) != expected_length:
            raise ValueError(f"Expected {expected_length} context freq tensors, got {len(freqs)}")
        return freqs

    @staticmethod
    def _concat_freqs(
        query_freqs_cis: Optional[torch.Tensor],
        context_freqs_cis: list[Optional[torch.Tensor]],
        include_query_in_kv: bool,
    ) -> Optional[torch.Tensor]:
        freq_parts = []
        if include_query_in_kv:
            freq_parts.append(query_freqs_cis)
        freq_parts.extend(context_freqs_cis)

        if not freq_parts:
            return None
        if any(freq is None for freq in freq_parts):
            if not all(freq is None for freq in freq_parts):
                raise ValueError("query/context freq tensors must be provided together")
            return None
        return torch.cat(freq_parts, dim=1) if len(freq_parts) > 1 else freq_parts[0]

    def contextual_forward(
        self,
        query_states: torch.Tensor,
        context_states: Optional[Sequence[torch.Tensor] | torch.Tensor] = None,
        query_freqs_cis: Optional[torch.Tensor] = None,
        context_freqs_cis: Optional[Sequence[torch.Tensor] | torch.Tensor] = None,
        attn_params: Optional[AttentionParams] = None,
        include_query_in_kv: bool = True,
    ) -> torch.Tensor:
        context_list = self._normalize_context_states(context_states)
        context_freqs_list = self._normalize_context_freqs(context_freqs_cis, len(context_list))

        kv_parts = [query_states] if include_query_in_kv else []
        kv_parts.extend(context_list)
        if not kv_parts:
            raise ValueError("contextual_forward requires query in kv or at least one context tensor")

        kv_states = torch.cat(kv_parts, dim=1) if len(kv_parts) > 1 else kv_parts[0]
        kv_freqs_cis = self._concat_freqs(query_freqs_cis, context_freqs_list, include_query_in_kv)

        query, gate_score, dtype = self._project_query(query_states, query_freqs_cis)
        key, value = self._project_key_value(kv_states, kv_freqs_cis, dtype=dtype)
        hidden_states = attention(query, key, value, attn_params=attn_params)
        return self._finalize_attention_output(hidden_states, gate_score, dtype)

    def _forward(
        self, hidden_states: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, attn_params: Optional[AttentionParams] = None
    ) -> torch.Tensor:
        query, gate_score, dtype = self._project_query(hidden_states, freqs_cis)
        key, value = self._project_key_value(hidden_states, freqs_cis, dtype=dtype)
        hidden_states = attention(query, key, value, attn_params=attn_params)
        return self._finalize_attention_output(hidden_states, gate_score, dtype)

    def forward(
        self, hidden_states: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, attn_params: Optional[AttentionParams] = None
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, hidden_states, freqs_cis, attn_params, use_reentrant=False)
        else:
            return self._forward(hidden_states, freqs_cis, attn_params)


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        use_16bit: bool = False,
        gate_type: str = "none",
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.modulation = modulation

        self.attention = ZImageAttention(
            dim,
            n_heads,
            n_kv_heads,
            qk_norm,
            norm_eps,
            use_16bit=use_16bit,
            gate_type=gate_type,
        )
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        if modulation:
            self.adaLN_modulation = nn.ModuleList([nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)])

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading
        self.feed_forward.enable_gradient_checkpointing()
        self.attention.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.feed_forward.disable_gradient_checkpointing()
        self.attention.disable_gradient_checkpointing()

    def set_log_gate_stats(self, enabled: bool):
        if hasattr(self.attention, "set_log_gate_stats"):
            self.attention.set_log_gate_stats(enabled)

    def set_gate_trainable(self, enabled: bool):
        if hasattr(self.attention, "set_gate_trainable"):
            self.attention.set_gate_trainable(enabled)

    def set_gate_enabled(self, enabled: bool):
        if hasattr(self.attention, "set_gate_enabled"):
            self.attention.set_gate_enabled(enabled)

    def get_gate_statistics(self) -> Dict[str, float]:
        if hasattr(self.attention, "get_gate_statistics"):
            return self.attention.get_gate_statistics()
        return {}

    def restore_frozen_gate(self):
        if hasattr(self.attention, "restore_frozen_gate"):
            self.attention.restore_frozen_gate()

    def _get_modulation(self, adaln_input: Optional[torch.Tensor]):
        if not self.modulation:
            return None
        assert adaln_input is not None
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation[0](adaln_input).unsqueeze(1).chunk(4, dim=2)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
        return scale_msa, gate_msa, scale_mlp, gate_mlp

    def _normalize_attention_input(self, x: torch.Tensor, scale_msa: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention_norm1(x)
        if scale_msa is not None:
            x = x * scale_msa
        return x

    def _apply_attention_and_ffn(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        scale_mlp: Optional[torch.Tensor] = None,
        gate_msa: Optional[torch.Tensor] = None,
        gate_mlp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if gate_msa is None:
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
            return x

        x = x + gate_msa * self.attention_norm2(attn_out)
        ffn_input = self.ffn_norm1(x)
        if scale_mlp is not None:
            ffn_input = ffn_input * scale_mlp
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(ffn_input))
        return x

    @staticmethod
    def _normalize_context_states(
        context_states: Optional[Sequence[torch.Tensor] | torch.Tensor],
    ) -> list[torch.Tensor]:
        if context_states is None:
            return []
        if isinstance(context_states, torch.Tensor):
            return [context_states]
        return list(context_states)

    @staticmethod
    def _normalize_context_freqs(
        context_freqs_cis: Optional[Sequence[torch.Tensor] | torch.Tensor],
        expected_length: int,
    ) -> list[Optional[torch.Tensor]]:
        if context_freqs_cis is None:
            return [None] * expected_length
        if isinstance(context_freqs_cis, torch.Tensor):
            freqs = [context_freqs_cis]
        else:
            freqs = list(context_freqs_cis)
        if len(freqs) != expected_length:
            raise ValueError(f"Expected {expected_length} context freq tensors, got {len(freqs)}")
        return freqs

    def contextual_forward(
        self,
        query_states: torch.Tensor,
        query_freqs_cis: Optional[torch.Tensor],
        context_states: Optional[Sequence[torch.Tensor] | torch.Tensor] = None,
        context_freqs_cis: Optional[Sequence[torch.Tensor] | torch.Tensor] = None,
        adaln_input: Optional[torch.Tensor] = None,
        attn_params: Optional[AttentionParams] = None,
        include_query_in_kv: bool = True,
    ) -> torch.Tensor:
        context_list = self._normalize_context_states(context_states)
        context_freqs_list = self._normalize_context_freqs(context_freqs_cis, len(context_list))

        modulation = self._get_modulation(adaln_input)
        if modulation is None:
            normalized_query = self._normalize_attention_input(query_states)
            normalized_context = [self._normalize_attention_input(context) for context in context_list]
            attn_out = self.attention.contextual_forward(
                normalized_query,
                context_states=normalized_context,
                query_freqs_cis=query_freqs_cis,
                context_freqs_cis=context_freqs_list,
                attn_params=attn_params,
                include_query_in_kv=include_query_in_kv,
            )
            return self._apply_attention_and_ffn(query_states, attn_out)

        scale_msa, gate_msa, scale_mlp, gate_mlp = modulation
        normalized_query = self._normalize_attention_input(query_states, scale_msa)
        normalized_context = [self._normalize_attention_input(context, scale_msa) for context in context_list]
        attn_out = self.attention.contextual_forward(
            normalized_query,
            context_states=normalized_context,
            query_freqs_cis=query_freqs_cis,
            context_freqs_cis=context_freqs_list,
            attn_params=attn_params,
            include_query_in_kv=include_query_in_kv,
        )
        return self._apply_attention_and_ffn(query_states, attn_out, scale_mlp=scale_mlp, gate_msa=gate_msa, gate_mlp=gate_mlp)

    def _forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        attn_params: Optional[AttentionParams] = None,
    ):
        modulation = self._get_modulation(adaln_input)
        if modulation is None:
            attn_out = self.attention(self._normalize_attention_input(x), freqs_cis=freqs_cis, attn_params=attn_params)
            return self._apply_attention_and_ffn(x, attn_out)

        scale_msa, gate_msa, scale_mlp, gate_mlp = modulation
        attn_out = self.attention(self._normalize_attention_input(x, scale_msa), freqs_cis=freqs_cis, attn_params=attn_params)
        return self._apply_attention_and_ffn(x, attn_out, scale_mlp=scale_mlp, gate_msa=gate_msa, gate_mlp=gate_mlp)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        attn_params: Optional[AttentionParams] = None,
    ):
        if self.training and self.gradient_checkpointing:
            forward_fn = self._forward
            if self.activation_cpu_offloading:
                forward_fn = create_cpu_offloading_wrapper(forward_fn, self.feed_forward.w1.weight.device)
            return checkpoint(forward_fn, x, freqs_cis, adaln_input, attn_params, use_reentrant=False)
        else:
            return self._forward(x, freqs_cis, adaln_input, attn_params)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(self, theta: float = ROPE_THETA, axes_dims: List[int] = ROPE_AXES_DIMS, axes_lens: List[int] = ROPE_AXES_LENS):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens)
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = ROPE_THETA):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
                freqs_cis.append(freqs_cis_i)
            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            # [torch.Size([1536, 16]), torch.Size([512, 24]), torch.Size([512, 24])]
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)  # keep on cpu

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i].to(device)[index])
        return torch.cat(result, dim=-1)


class ZImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=ROPE_THETA,
        t_scale=1000.0,
        axes_dims=ROPE_AXES_DIMS,
        axes_lens=ROPE_AXES_LENS,
        attn_mode: str = "torch",
        split_attn: bool = False,
        use_16bit_for_attention: bool = False,
        gate_type: str = "none",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.attn_mode = attn_mode
        self.split_attn = split_attn
        self.gate_type = gate_type

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder
            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)

        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                    use_16bit=use_16bit_for_attention,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                    use_16bit=use_16bit_for_attention,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    use_16bit=use_16bit_for_attention,
                    gate_type=gate_type,
                )
                for layer_id in range(n_layers)
            ]
        )

        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = None

        self.offloader = None
        self.num_blocks = n_layers

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = cpu_offload

        for block in self.noise_refiner + self.context_refiner + self.layers:
            block.enable_gradient_checkpointing(activation_cpu_offloading=cpu_offload)

        print(f"Z-Image: Gradient checkpointing enabled. CPU offload: {cpu_offload}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

        for block in self.noise_refiner + self.context_refiner + self.layers:
            block.disable_gradient_checkpointing()

        print("Z-Image: Gradient checkpointing disabled.")

    def set_log_gate_stats(self, enabled: bool):
        for block in self.noise_refiner + self.context_refiner + self.layers:
            if hasattr(block, "set_log_gate_stats"):
                block.set_log_gate_stats(enabled)

    def set_gate_trainable(self, enabled: bool):
        for block in self.noise_refiner + self.context_refiner + self.layers:
            if hasattr(block, "set_gate_trainable"):
                block.set_gate_trainable(enabled)

    def set_gate_layers(
        self,
        layer_ids: Optional[List[int]] = None,
        noise_refiner_ids: Optional[List[int]] = None,
        context_refiner_ids: Optional[List[int]] = None,
    ):
        def apply(blocks, enabled_ids):
            if enabled_ids is None:
                return
            enabled = set(enabled_ids)
            for idx, block in enumerate(blocks):
                if hasattr(block, "set_gate_enabled"):
                    block.set_gate_enabled(idx in enabled)

        apply(self.layers, layer_ids)
        if noise_refiner_ids is not None or context_refiner_ids is not None:
            logger.warning("Refiner gate layer masks are ignored: gated attention is only applied to main transformer layers.")

    def get_gate_statistics(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        gate_means: List[float] = []
        gate_sparsities: List[float] = []

        for prefix, blocks in (
            ("noise_refiner", self.noise_refiner),
            ("context_refiner", self.context_refiner),
            ("layer", self.layers),
        ):
            for idx, block in enumerate(blocks):
                if not hasattr(block, "get_gate_statistics"):
                    continue
                block_stats = block.get_gate_statistics()
                for stat_name, stat_value in block_stats.items():
                    key = f"{prefix}_{idx}/{stat_name}"
                    stats[key] = stat_value
                    if stat_name == "gate_mean":
                        gate_means.append(stat_value)
                    elif stat_name == "gate_sparsity":
                        gate_sparsities.append(stat_value)

        if gate_means:
            stats["gate_mean_overall"] = sum(gate_means) / len(gate_means)
        if gate_sparsities:
            stats["gate_sparsity_overall"] = sum(gate_sparsities) / len(gate_sparsities)

        return stats

    def restore_frozen_gates(self):
        for block in self.noise_refiner + self.context_refiner + self.layers:
            if hasattr(block, "restore_frozen_gate"):
                block.restore_frozen_gate()

    def enable_block_swap(self, num_blocks: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False):
        self.blocks_to_swap = num_blocks

        assert self.blocks_to_swap <= self.num_blocks - 2, (
            f"Cannot swap more than {self.num_blocks - 2} double blocks. Requested {self.blocks_to_swap} double blocks."
        )

        self.offloader = ModelOffloader(
            "double", self.layers, len(self.layers), self.blocks_to_swap, supports_backward, device, use_pinned_memory
        )
        print(
            f"Z-Image: Block swap enabled. Swapping {num_blocks} of {self.num_blocks} blocks to device {device}. Supports backward: {supports_backward}"
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print("Z-Image: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print("Z-Image: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
        if self.blocks_to_swap:
            save_layers = self.layers
            self.layers = nn.ModuleList()

        self.to(device)

        if self.blocks_to_swap:
            self.layers = save_layers

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.layers)

    def prepare_adaln_input(self, t: torch.Tensor, reference_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        t = t * self.t_scale
        adaln_input = self.t_embedder(t)
        if reference_dtype is not None:
            adaln_input = adaln_input.to(reference_dtype)
        return adaln_input

    def unpatchify(self, x: torch.Tensor, size: Tuple[int, int, int], patch_size: int, f_patch_size: int) -> torch.Tensor:
        """
        Unpatchify the latent tensor back to image/video format.

        Args:
            x: [B, seq_len, patch_dim] tensor
            size: (F, H, W) tuple of the original latent size
            patch_size: spatial patch size (pH = pW)
            f_patch_size: temporal patch size (pF)

        Returns:
            [B, C, F, H, W] tensor
        """
        pH = pW = patch_size
        pF = f_patch_size
        F_size, H_size, W_size = size
        B = x.shape[0]
        F_tokens, H_tokens, W_tokens = F_size // pF, H_size // pH, W_size // pW
        ori_len = F_tokens * H_tokens * W_tokens

        # Take only the original image tokens (exclude caption part if any)
        x = x[:, :ori_len]  # [B, ori_len, patch_dim]

        x = x.view(B, F_tokens, H_tokens, W_tokens, pF, pH, pW, self.out_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # [B, C, F_tokens, pF, H_tokens, pH, W_tokens, pW]
        x = x.reshape(B, self.out_channels, F_size, H_size, W_size)
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify(self, x: torch.Tensor, patch_size: int, f_patch_size: int) -> torch.Tensor:
        """
        Patchify the latent tensor.

        Args:
            x: [B, C, F, H, W] tensor
            patch_size: spatial patch size (pH = pW)
            f_patch_size: temporal patch size (pF)

        Returns:
            [B, seq_len, patch_dim] tensor where seq_len = (F/pF) * (H/pH) * (W/pW)
            and patch_dim = pF * pH * pW * C
        """
        pH = pW = patch_size
        pF = f_patch_size
        B, C, F_size, H_size, W_size = x.shape
        F_tokens, H_tokens, W_tokens = F_size // pF, H_size // pH, W_size // pW

        x = x.view(B, C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)  # [B, F_tokens, H_tokens, W_tokens, pF, pH, pW, C]
        x = x.reshape(B, F_tokens * H_tokens * W_tokens, pF * pH * pW * C)
        return x

    def create_image_position_ids(
        self, F_tokens: int, H_tokens: int, W_tokens: int, cap_seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create position IDs for image patches.

        Args:
            F_tokens: number of frame tokens
            H_tokens: number of height tokens
            W_tokens: number of width tokens
            cap_seq_len: caption sequence length (for offset)
            device: device to create tensor on

        Returns:
            [seq_len, 3] tensor of position IDs
        """
        # Image positions start after caption positions
        # Position format: (cap_seq_len + 1 + f_idx, h_idx, w_idx). [F_tokens * H_tokens * W_tokens, 3]
        return self.create_coordinate_grid(
            size=(F_tokens, H_tokens, W_tokens), start=(cap_seq_len + 1, 0, 0), device=device
        ).flatten(0, 2)

    def create_caption_position_ids(self, cap_seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create position IDs for caption tokens.

        Args:
            cap_seq_len: caption sequence length
            device: device to create tensor on

        Returns:
            [cap_seq_len, 3] tensor of position IDs
        """
        # Caption positions: (i + 1, 0, 0) for i in range(cap_seq_len). [cap_seq_len, 3]
        return self.create_coordinate_grid(size=(cap_seq_len, 1, 1), start=(1, 0, 0), device=device).flatten(0, 2)

    def prepare_image_tokens(
        self,
        x: torch.Tensor,
        cap_seq_len: int,
        patch_size: int = 2,
        f_patch_size: int = 1,
        adaln_input: Optional[torch.Tensor] = None,
        apply_noise_refiner: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        B, C, F_size, H_size, W_size = x.shape
        device = x.device

        pH = pW = patch_size
        pF = f_patch_size
        F_tokens, H_tokens, W_tokens = F_size // pF, H_size // pH, W_size // pW
        x_seq_len = F_tokens * H_tokens * W_tokens

        x_tokens = self.patchify(x, patch_size, f_patch_size)
        x_tokens = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x_tokens)

        x_pos_ids = self.create_image_position_ids(F_tokens, H_tokens, W_tokens, cap_seq_len, device)
        x_freqs_cis = self.rope_embedder(x_pos_ids)
        x_freqs_cis = x_freqs_cis.unsqueeze(0).expand(B, -1, -1)

        if apply_noise_refiner:
            if adaln_input is None:
                raise ValueError("adaln_input is required when apply_noise_refiner=True")
            adaln_input = adaln_input.type_as(x_tokens)
            noise_refiner_attn_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, 0, None)
            for layer in self.noise_refiner:
                x_tokens = layer(x_tokens, x_freqs_cis, adaln_input, attn_params=noise_refiner_attn_params)

        metadata = {
            "image_shape": (F_size, H_size, W_size),
            "token_shape": (F_tokens, H_tokens, W_tokens),
            "seq_len": x_seq_len,
            "patch_size": patch_size,
            "f_patch_size": f_patch_size,
        }
        return x_tokens, x_freqs_cis, metadata

    def prepare_caption_tokens(
        self,
        cap_feats: torch.Tensor,
        cap_mask: Optional[torch.Tensor],
        apply_context_refiner: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = cap_feats.device
        cap_seq_len = cap_feats.shape[1]

        cap_feats = self.cap_embedder(cap_feats)
        if cap_mask is not None:
            cap_pad_mask = ~cap_mask
            cap_feats = cap_feats.masked_fill(cap_pad_mask.unsqueeze(-1), 0.0)
            cap_feats = cap_feats + self.cap_pad_token * cap_pad_mask.unsqueeze(-1).to(dtype=cap_feats.dtype)

        cap_pos_ids = self.create_caption_position_ids(cap_seq_len, device)
        cap_freqs_cis = self.rope_embedder(cap_pos_ids)
        cap_freqs_cis = cap_freqs_cis.unsqueeze(0).expand(cap_feats.shape[0], -1, -1)

        if apply_context_refiner:
            context_refiner_attn_params = AttentionParams.create_attention_params_from_mask(
                self.attn_mode, self.split_attn, 0, cap_mask
            )
            for layer in self.context_refiner:
                cap_feats = layer(cap_feats, cap_freqs_cis, attn_params=context_refiner_attn_params)

        return cap_feats, cap_freqs_cis

    @staticmethod
    def build_unified_tokens(
        x_tokens: torch.Tensor,
        x_freqs_cis: torch.Tensor,
        cap_tokens: torch.Tensor,
        cap_freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unified = torch.cat([x_tokens, cap_tokens], dim=1)
        unified_freqs_cis = torch.cat([x_freqs_cis, cap_freqs_cis], dim=1)
        return unified, unified_freqs_cis

    @staticmethod
    def split_unified_tokens(unified: torch.Tensor, x_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return unified[:, :x_seq_len], unified[:, x_seq_len:]

    @staticmethod
    def _coerce_token_indices(token_indices: Sequence[int] | torch.Tensor, device: torch.device) -> torch.Tensor:
        if isinstance(token_indices, torch.Tensor):
            indices = token_indices.to(device=device, dtype=torch.long)
        else:
            indices = torch.tensor(list(token_indices), device=device, dtype=torch.long)
        return indices

    def select_token_subset(
        self,
        tokens: torch.Tensor,
        token_indices: Sequence[int] | torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        indices = self._coerce_token_indices(token_indices, tokens.device)
        subset_tokens = tokens.index_select(1, indices)
        if freqs_cis is None:
            return subset_tokens, None
        subset_freqs = freqs_cis.index_select(1, indices)
        return subset_tokens, subset_freqs

    def replace_token_subset(
        self,
        tokens: torch.Tensor,
        token_indices: Sequence[int] | torch.Tensor,
        replacement_tokens: torch.Tensor,
    ) -> torch.Tensor:
        indices = self._coerce_token_indices(token_indices, tokens.device)
        updated = tokens.clone()
        updated.index_copy_(1, indices, replacement_tokens)
        return updated

    def create_main_attention_params(self, x_seq_len: int, cap_mask: Optional[torch.Tensor]) -> AttentionParams:
        return AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, x_seq_len, cap_mask)

    def run_main_layers(
        self,
        unified: torch.Tensor,
        unified_freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor,
        cap_mask: Optional[torch.Tensor],
        x_seq_len: int,
        start_layer: int = 0,
        end_layer: Optional[int] = None,
    ) -> torch.Tensor:
        if end_layer is None:
            end_layer = len(self.layers)

        attn_params = self.create_main_attention_params(x_seq_len, cap_mask)
        for index in range(start_layer, end_layer):
            layer = self.layers[index]
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)

            unified = layer(unified, unified_freqs_cis, adaln_input, attn_params=attn_params)

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.layers, index)

        return unified

    def finalize_image_tokens(
        self,
        unified: torch.Tensor,
        adaln_input: torch.Tensor,
        image_size: tuple[int, int, int],
        patch_size: int = 2,
        f_patch_size: int = 1,
    ) -> torch.Tensor:
        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        return self.unpatchify(unified, image_size, patch_size, f_patch_size)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cap_feats: torch.Tensor,
        cap_mask: torch.Tensor,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ) -> torch.Tensor:
        """
        Forward pass of the Z-Image Transformer.

        Args:
            x: Latent tensor [B, C, F, H, W]
            t: Timestep tensor [B]
            cap_feats: Caption features [B, cap_seq_len, cap_feat_dim]
            cap_mask: Caption mask [B, cap_seq_len], True for valid tokens
            patch_size: Spatial patch size (default: 2)
            f_patch_size: Temporal patch size (default: 1)

        Returns:
            Output tensor [B, C, F, H, W]
        """
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        B, C, F_size, H_size, W_size = x.shape
        device = x.device
        cap_seq_len = cap_feats.shape[1]

        adaln_input = self.prepare_adaln_input(t)
        x_tokens, x_freqs_cis, x_meta = self.prepare_image_tokens(
            x,
            cap_seq_len=cap_seq_len,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
            adaln_input=adaln_input,
        )
        adaln_input = adaln_input.type_as(x_tokens)
        cap_tokens, cap_freqs_cis = self.prepare_caption_tokens(cap_feats, cap_mask)
        unified, unified_freqs_cis = self.build_unified_tokens(x_tokens, x_freqs_cis, cap_tokens, cap_freqs_cis)

        x_seq_len = x_meta["seq_len"]
        unified = self.run_main_layers(unified, unified_freqs_cis, adaln_input, cap_mask, x_seq_len)

        unified = unified.to(device)  # ensure unified is on the correct device when activation CPU offloading is used

        x = self.finalize_image_tokens(unified, adaln_input, (F_size, H_size, W_size), patch_size, f_patch_size)

        return x




def create_model(attn_mode: str, split_attn: bool, dtype: Optional[torch.dtype], gate_type: str = "none") -> ZImageTransformer2DModel:
    with init_empty_weights():
        logger.info("Creating ZImageTransformer2DModel")
        model = ZImageTransformer2DModel(
            all_patch_size=tuple(zimage_config.DEFAULT_TRANSFORMER_PATCH_SIZE),
            all_f_patch_size=tuple(zimage_config.DEFAULT_TRANSFORMER_F_PATCH_SIZE),
            in_channels=zimage_config.DEFAULT_TRANSFORMER_IN_CHANNELS,
            dim=zimage_config.DEFAULT_TRANSFORMER_DIM,
            n_layers=zimage_config.DEFAULT_TRANSFORMER_N_LAYERS,
            n_refiner_layers=zimage_config.DEFAULT_TRANSFORMER_N_REFINER_LAYERS,
            n_heads=zimage_config.DEFAULT_TRANSFORMER_N_HEADS,
            n_kv_heads=zimage_config.DEFAULT_TRANSFORMER_N_KV_HEADS,
            norm_eps=zimage_config.DEFAULT_TRANSFORMER_NORM_EPS,
            qk_norm=zimage_config.DEFAULT_TRANSFORMER_QK_NORM,
            cap_feat_dim=zimage_config.DEFAULT_TRANSFORMER_CAP_FEAT_DIM,
            rope_theta=zimage_config.ROPE_THETA,
            t_scale=zimage_config.DEFAULT_TRANSFORMER_T_SCALE,
            axes_dims=zimage_config.ROPE_AXES_DIMS,
            axes_lens=zimage_config.ROPE_AXES_LENS,
            attn_mode=attn_mode,
            split_attn=split_attn,
            gate_type=gate_type,
        )
        if dtype is not None:
            model.to(dtype)
    return model


def _expand_split_files(file_path: str) -> list[str]:
    import os
    import re

    basename = os.path.basename(file_path)
    match = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", basename)
    if not match:
        return [file_path]

    prefix = basename[: match.start(2)]
    count = int(match.group(3))
    files = []
    for i in range(count):
        name = f"{prefix}{i + 1:05d}-of-{count:05d}.safetensors"
        files.append(os.path.join(os.path.dirname(file_path), name))
    return files


def _find_weight_files(path: str) -> list[str]:
    import glob
    import os

    if os.path.isdir(path):
        for candidate in (os.path.join(path, "transformer"), path):
            if not os.path.isdir(candidate):
                continue
            files = sorted(glob.glob(os.path.join(candidate, "*.safetensors")))
            if files:
                return files
        raise FileNotFoundError(f"No .safetensors found under {path}")

    return _expand_split_files(path)


def _convert_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    replace_keys = {
        "all_final_layer.2-1.linear": "final_layer.linear",
        "all_final_layer.2-1.adaLN_modulation": "final_layer.adaLN_modulation",
        "all_x_embedder.2-1.bias": "x_embedder.bias",
        "all_x_embedder.2-1.weight": "x_embedder.weight",
        ".attention.to_out.0.bias": ".attention.out.bias",
        ".attention.norm_k.weight": ".attention.k_norm.weight",
        ".attention.norm_q.weight": ".attention.q_norm.weight",
        ".attention.to_out.0.weight": ".attention.out.weight",
    }
    replace_keys_reverse = {v: k for k, v in replace_keys.items()}

    new_sd: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model.diffusion_model."):
            key = key.replace("model.diffusion_model.", "", 1)

        for k, v in replace_keys_reverse.items():
            if k in key:
                key = key.replace(k, v)
                break

        if "attention.qkv.weight" in key:
            for new_key, chunk in zip(
                [key.replace("qkv", "to_q"), key.replace("qkv", "to_k"), key.replace("qkv", "to_v")],
                torch.chunk(value, 3, dim=0),
            ):
                new_sd[new_key] = chunk
            continue

        if "attention.qkv.bias" in key:
            for new_key, chunk in zip(
                [key.replace("qkv", "to_q"), key.replace("qkv", "to_k"), key.replace("qkv", "to_v")],
                torch.chunk(value, 3, dim=0),
            ):
                new_sd[new_key] = chunk
            continue

        new_sd[key] = value

    return new_sd


def _expand_state_dict_for_gating(
    state_dict: Dict[str, torch.Tensor],
    gate_type: str,
    n_heads: int,
    dim: int,
) -> Dict[str, torch.Tensor]:
    if gate_type == "none":
        gate_dim = 0
    elif gate_type == "headwise":
        gate_dim = n_heads
    elif gate_type == "elementwise":
        gate_dim = dim
    else:
        raise ValueError(f"Unsupported gate_type: {gate_type}")

    base_out = dim
    new_sd: Dict[str, torch.Tensor] = {}

    def expected_out_features(key: str) -> int:
        # Gated attention is only used in the main transformer layers.
        return dim + gate_dim if gate_dim > 0 and key.startswith("layers.") else dim

    for key, value in state_dict.items():
        if ".attention.to_q.weight" in key:
            expected_out = expected_out_features(key)
            if value.shape[0] == expected_out:
                new_sd[key] = value
            elif value.shape[0] == base_out and expected_out > base_out:
                new_weight = torch.zeros((expected_out, value.shape[1]), dtype=value.dtype, device=value.device)
                new_weight[:base_out] = value
                new_sd[key] = new_weight
            elif value.shape[0] > expected_out and value.shape[1] == dim:
                logger.info(f"Trim extra gated to_q rows for {key}: {value.shape[0]} -> {expected_out}")
                new_sd[key] = value[:expected_out]
            else:
                raise ValueError(
                    f"Unexpected to_q.weight shape for {key}: {tuple(value.shape)} "
                    f"(expected {expected_out} rows, base {base_out})"
                )
        elif ".attention.to_q.bias" in key:
            expected_out = expected_out_features(key)
            if value.shape[0] == expected_out:
                new_sd[key] = value
            elif value.shape[0] == base_out and expected_out > base_out:
                new_bias = torch.zeros(expected_out, dtype=value.dtype, device=value.device)
                new_bias[:base_out] = value
                new_sd[key] = new_bias
            elif value.shape[0] > expected_out:
                logger.info(f"Trim extra gated to_q bias rows for {key}: {value.shape[0]} -> {expected_out}")
                new_sd[key] = value[:expected_out]
            else:
                raise ValueError(
                    f"Unexpected to_q.bias shape for {key}: {tuple(value.shape)} "
                    f"(expected {expected_out}, base {base_out})"
                )
        else:
            new_sd[key] = value

    return new_sd


def load_zimage_model(
    dit_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    attn_mode: str = "torch",
    split_attn: bool = False,
    disable_mmap: bool = False,
    gate_type: str = "none",
) -> ZImageTransformer2DModel:
    from library.utils import load_safetensors

    device = torch.device(device)
    model = create_model(attn_mode, split_attn, dtype, gate_type=gate_type)

    weight_files = _find_weight_files(dit_path)
    logger.info(f"Loading DiT weights from: {weight_files}")

    sd: Dict[str, torch.Tensor] = {}
    for file_path in weight_files:
        sd.update(load_safetensors(file_path, device="cpu", disable_mmap=disable_mmap, dtype=None))

    sd = _convert_state_dict_keys(sd)
    sd = _expand_state_dict_for_gating(sd, gate_type, model.n_heads, model.dim)

    info = model.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded DiT weights, info={info}")

    model.to(device)
    if dtype is not None:
        model.to(dtype)

    return model
