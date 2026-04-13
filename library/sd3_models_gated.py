# Gated Attention for SD3.5 MMDiT
# Based on the paper "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"
# https://arxiv.org/abs/2505.06708

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sd3_models import (
    RMSNorm,
    MLP,
    SwiGLUFeedForward,
    MEMORY_LAYOUTS,
    modulate,
    attention,
    memory_efficient_attention,
)

from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class GatedAttentionLinears(nn.Module):
    """
    Attention Linears with Gated Attention mechanism.

    Supports two gating modes:
    - headwise: Each attention head has a scalar gate
    - elementwise: Each element of the attention output has a gate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        pre_only: bool = False,
        qk_norm: Optional[str] = None,
        gate_type: str = "headwise",  # "headwise" or "elementwise" or "none"
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.gate_type = gate_type
        self.gate_enabled = True

        # Calculate additional dimensions for gate
        if gate_type == "headwise":
            # For headwise gating, we add num_heads scalar gates
            gate_dim = num_heads
        elif gate_type == "elementwise":
            # For elementwise gating, we add dim gates (same as query dimension)
            gate_dim = dim
        else:
            gate_dim = 0

        # Expand qkv to include gate values
        # Original: dim * 3 for q, k, v
        # With gate: dim * 3 + gate_dim for q, k, v, and gate
        self.qkv = nn.Linear(dim, dim * 3 + gate_dim, bias=qkv_bias)
        self.gate_dim = gate_dim

        if not pre_only:
            self.proj = nn.Linear(dim, dim)
        self.pre_only = pre_only

        if qk_norm == "rms":
            self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
            self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
        elif qk_norm == "ln":
            self.ln_q = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
            self.ln_k = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
        elif qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise ValueError(qk_norm)

    def pre_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        output:
            q, k, v: [B, L, D]
            gate_score: [B, L, num_heads, 1] for headwise or [B, L, num_heads, head_dim] for elementwise
        """
        B, L, C = x.shape
        qkv_gate: torch.Tensor = self.qkv(x)

        if self.gate_type == "headwise":
            # Split: q (dim), k (dim), v (dim), gate (num_heads)
            qkv, gate_score = torch.split(qkv_gate, [C * 3, self.gate_dim], dim=-1)
            q, k, v = qkv.reshape(B, L, -1, self.head_dim).chunk(3, dim=2)
            # Reshape gate_score to [B, L, num_heads, 1]
            gate_score = gate_score.reshape(B, L, self.num_heads, 1)
        elif self.gate_type == "elementwise":
            # Split: q (dim), k (dim), v (dim), gate (dim)
            qkv, gate_score = torch.split(qkv_gate, [C * 3, self.gate_dim], dim=-1)
            q, k, v = qkv.reshape(B, L, -1, self.head_dim).chunk(3, dim=2)
            # Reshape gate_score to [B, L, num_heads, head_dim]
            gate_score = gate_score.reshape(B, L, self.num_heads, self.head_dim)
        else:
            qkv = qkv_gate
            q, k, v = qkv.reshape(B, L, -1, self.head_dim).chunk(3, dim=2)
            gate_score = None

        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        v = v.reshape(v.shape[0], v.shape[1], -1)

        return q, k, v, gate_score

    def set_gate_enabled(self, enabled: bool):
        self.gate_enabled = enabled

    def apply_gate(self, attn_output: torch.Tensor, gate_score: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply gating to attention output.

        Args:
            attn_output: [B, L, num_heads, head_dim] or [B, L, D]
            gate_score: [B, L, num_heads, 1] for headwise or [B, L, num_heads, head_dim] for elementwise

        Returns:
            Gated attention output with same shape as input
        """
        if gate_score is None or self.gate_type == "none" or not self.gate_enabled:
            return attn_output

        B, L = attn_output.shape[:2]

        # Reshape attn_output to [B, L, num_heads, head_dim] if needed
        if attn_output.dim() == 3:
            attn_output = attn_output.reshape(B, L, self.num_heads, self.head_dim)

        # Apply sigmoid gate
        gated_output = attn_output * torch.sigmoid(gate_score)

        # Reshape back to [B, L, D]
        return gated_output.reshape(B, L, -1)

    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        x = self.proj(x)
        return x

    def get_gate_statistics(self, gate_score: Optional[torch.Tensor]) -> Dict[str, float]:
        """
        Get statistics about the gate values for tensorboard logging.

        Returns:
            Dictionary with gate statistics (mean, std, sparsity)
        """
        if gate_score is None:
            return {}

        gate_values = torch.sigmoid(gate_score)

        stats = {
            "gate_mean": gate_values.mean().item(),
            "gate_std": gate_values.std().item(),
            "gate_min": gate_values.min().item(),
            "gate_max": gate_values.max().item(),
            # Sparsity: percentage of gates below 0.5
            "gate_sparsity": (gate_values < 0.5).float().mean().item(),
        }

        return stats


class GatedSingleDiTBlock(nn.Module):
    """
    A DiT block with gated adaptive layer norm (adaLN) conditioning and gated attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = "xformers",
        qkv_bias: bool = False,
        pre_only: bool = False,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: Optional[str] = None,
        x_block_self_attn: bool = False,
        gate_type: str = "headwise",  # "headwise" or "elementwise" or "none"
        **block_kwargs,
    ):
        super().__init__()
        assert attn_mode in MEMORY_LAYOUTS
        self.attn_mode = attn_mode
        self.gate_type = gate_type

        if not rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = GatedAttentionLinears(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias,
            pre_only=pre_only, qk_norm=qk_norm, gate_type=gate_type
        )

        self.x_block_self_attn = x_block_self_attn
        if self.x_block_self_attn:
            assert not pre_only
            assert not scale_mod_only
            self.attn2 = GatedAttentionLinears(
                dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias,
                pre_only=False, qk_norm=qk_norm, gate_type=gate_type
            )

        if not pre_only:
            if not rmsnorm:
                self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            else:
                self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not pre_only:
            if not swiglu:
                self.mlp = MLP(
                    in_features=hidden_size,
                    hidden_features=mlp_hidden_dim,
                    act_layer=lambda: nn.GELU(approximate="tanh"),
                )
            else:
                self.mlp = SwiGLUFeedForward(
                    dim=hidden_size,
                    hidden_dim=mlp_hidden_dim,
                    multiple_of=256,
                )

        self.scale_mod_only = scale_mod_only
        if self.x_block_self_attn:
            n_mods = 9
        elif not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size))
        self.pre_only = pre_only

        # Store gate statistics for logging (not the tensors themselves to avoid memory issues)
        self._last_gate_stats = None
        self._last_gate_stats2 = None
        self._log_gate_stats = False  # Flag to control logging

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor) -> Tuple:
        if not self.pre_only:
            if not self.scale_mod_only:
                (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=-1)
            else:
                shift_msa = None
                shift_mlp = None
                (scale_msa, gate_msa, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(4, dim=-1)
            q, k, v, gate_score = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            # Store statistics only, not the tensor itself (to save memory during gradient checkpointing)
            if self._log_gate_stats and gate_score is not None:
                with torch.no_grad():
                    self._last_gate_stats = self.attn.get_gate_statistics(gate_score)
            return (q, k, v, gate_score), (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            if not self.scale_mod_only:
                (shift_msa, scale_msa) = self.adaLN_modulation(c).chunk(2, dim=-1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)
            q, k, v, gate_score = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            if self._log_gate_stats and gate_score is not None:
                with torch.no_grad():
                    self._last_gate_stats = self.attn.get_gate_statistics(gate_score)
            return (q, k, v, gate_score), None

    def pre_attention_x(self, x: torch.Tensor, c: torch.Tensor) -> Tuple:
        assert self.x_block_self_attn
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2) = self.adaLN_modulation(
            c
        ).chunk(9, dim=1)
        x_norm = self.norm1(x)
        q, k, v, gate_score = self.attn.pre_attention(modulate(x_norm, shift_msa, scale_msa))
        q2, k2, v2, gate_score2 = self.attn2.pre_attention(modulate(x_norm, shift_msa2, scale_msa2))
        # Store statistics only
        if self._log_gate_stats:
            with torch.no_grad():
                if gate_score is not None:
                    self._last_gate_stats = self.attn.get_gate_statistics(gate_score)
                if gate_score2 is not None:
                    self._last_gate_stats2 = self.attn2.get_gate_statistics(gate_score2)
        return (q, k, v, gate_score), (q2, k2, v2, gate_score2), (x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2)

    def post_attention(self, attn, gate_score, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        assert not self.pre_only
        # Apply gating to attention output
        attn = self.attn.apply_gate(attn, gate_score)
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def post_attention_x(self, attn, gate_score, attn2, gate_score2, x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2, attn1_dropout: float = 0.0):
        assert not self.pre_only
        # Apply gating to attention outputs
        attn = self.attn.apply_gate(attn, gate_score)
        attn2 = self.attn2.apply_gate(attn2, gate_score2)

        if attn1_dropout > 0.0:
            attn1_dropout = torch.bernoulli(torch.full((attn.size(0), 1, 1), 1 - attn1_dropout, device=attn.device))
            attn_ = gate_msa.unsqueeze(1) * self.attn.post_attention(attn) * attn1_dropout
        else:
            attn_ = gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + attn_
        attn2_ = gate_msa2.unsqueeze(1) * self.attn2.post_attention(attn2)
        x = x + attn2_
        mlp_ = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + mlp_
        return x

    def set_gate_enabled(self, enabled: bool):
        self.attn.set_gate_enabled(enabled)
        if self.x_block_self_attn:
            self.attn2.set_gate_enabled(enabled)

    def set_log_gate_stats(self, enabled: bool):
        """Enable or disable gate statistics logging."""
        self._log_gate_stats = enabled

    def get_gate_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get gate statistics for tensorboard logging."""
        stats = {}
        if self._last_gate_stats is not None:
            stats["attn1"] = self._last_gate_stats
        if self._last_gate_stats2 is not None:
            stats["attn2"] = self._last_gate_stats2
        return stats


class GatedMMDiTBlock(nn.Module):
    """MMDiT Block with Gated Attention."""

    def __init__(self, *args, gate_type: str = "headwise", **kwargs):
        super().__init__()
        pre_only = kwargs.pop("pre_only")
        x_block_self_attn = kwargs.pop("x_block_self_attn")

        self.context_block = GatedSingleDiTBlock(
            *args, pre_only=pre_only, gate_type=gate_type, **kwargs
        )
        self.x_block = GatedSingleDiTBlock(
            *args, pre_only=False, x_block_self_attn=x_block_self_attn, gate_type=gate_type, **kwargs
        )

        self.head_dim = self.x_block.attn.head_dim
        self.mode = self.x_block.attn_mode
        self.gradient_checkpointing = False
        self.gate_type = gate_type

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def set_gate_enabled(self, enabled: bool):
        self.context_block.set_gate_enabled(enabled)
        self.x_block.set_gate_enabled(enabled)

    def set_log_gate_stats(self, enabled: bool):
        """Enable or disable gate statistics logging."""
        self.context_block.set_log_gate_stats(enabled)
        self.x_block.set_log_gate_stats(enabled)

    def _forward(self, context, x, c):
        ctx_qkv_gate, ctx_intermediate = self.context_block.pre_attention(context, c)
        ctx_q, ctx_k, ctx_v, ctx_gate_score = ctx_qkv_gate

        if self.x_block.x_block_self_attn:
            x_qkv_gate, x_qkv_gate2, x_intermediates = self.x_block.pre_attention_x(x, c)
            x_q, x_k, x_v, x_gate_score = x_qkv_gate
            x_q2, x_k2, x_v2, x_gate_score2 = x_qkv_gate2
        else:
            x_qkv_gate, x_intermediates = self.x_block.pre_attention(x, c)
            x_q, x_k, x_v, x_gate_score = x_qkv_gate

        ctx_len = ctx_q.size(1)

        q = torch.concat((ctx_q, x_q), dim=1)
        k = torch.concat((ctx_k, x_k), dim=1)
        v = torch.concat((ctx_v, x_v), dim=1)

        attn = attention(q, k, v, head_dim=self.head_dim, mode=self.mode)
        ctx_attn_out = attn[:, :ctx_len]
        x_attn_out = attn[:, ctx_len:]

        if self.x_block.x_block_self_attn:
            attn2 = attention(x_q2, x_k2, x_v2, self.x_block.attn2.num_heads, mode=self.mode)
            x = self.x_block.post_attention_x(x_attn_out, x_gate_score, attn2, x_gate_score2, *x_intermediates)
        else:
            x = self.x_block.post_attention(x_attn_out, x_gate_score, *x_intermediates)

        if not self.context_block.pre_only:
            context = self.context_block.post_attention(ctx_attn_out, ctx_gate_score, *ctx_intermediate)
        else:
            context = None

        return context, x

    def forward(self, *args, **kwargs):
        if self.training and self.gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def get_gate_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get gate statistics for tensorboard logging."""
        stats = {
            "context_block": self.context_block.get_gate_statistics(),
            "x_block": self.x_block.get_gate_statistics(),
        }
        return stats


def convert_attention_linears_to_gated(
    state_dict: Dict[str, torch.Tensor],
    gate_type: str = "headwise",
    num_heads: int = 38,  # SD3.5 Medium has 38 heads (depth=38)
) -> Dict[str, torch.Tensor]:
    """
    Convert original AttentionLinears weights to GatedAttentionLinears format.

    The original qkv weight has shape [dim * 3, dim].
    The new qkv weight has shape [dim * 3 + gate_dim, dim].

    We initialize the gate weights to zeros so that sigmoid(0) = 0.5,
    which means the model starts with neutral gating.
    """
    new_state_dict = {}
    head_dim = None

    for key, value in state_dict.items():
        if "attn.qkv.weight" in key or "attn2.qkv.weight" in key:
            # This is the qkv weight, need to expand it
            dim = value.shape[1]  # input dimension
            out_dim = value.shape[0]  # should be dim * 3

            if head_dim is None:
                # Calculate head_dim from the dimensions
                # dim * 3 = out_dim, so dim = out_dim / 3
                hidden_dim = out_dim // 3
                head_dim = hidden_dim // num_heads

            # Calculate gate dimension
            if gate_type == "headwise":
                gate_dim = num_heads
            elif gate_type == "elementwise":
                gate_dim = out_dim // 3  # same as hidden_dim
            else:
                gate_dim = 0

            if gate_dim > 0:
                # Create new weight with additional gate dimensions
                new_weight = torch.zeros(out_dim + gate_dim, dim, dtype=value.dtype, device=value.device)
                new_weight[:out_dim] = value
                # Gate weights are initialized to 0, so sigmoid(0) = 0.5
                new_state_dict[key] = new_weight
            else:
                new_state_dict[key] = value

        elif "attn.qkv.bias" in key or "attn2.qkv.bias" in key:
            # This is the qkv bias, need to expand it too
            out_dim = value.shape[0]  # should be dim * 3
            hidden_dim = out_dim // 3

            # Calculate gate dimension
            if gate_type == "headwise":
                gate_dim = num_heads
            elif gate_type == "elementwise":
                gate_dim = hidden_dim
            else:
                gate_dim = 0

            if gate_dim > 0:
                # Create new bias with additional gate dimensions
                new_bias = torch.zeros(out_dim + gate_dim, dtype=value.dtype, device=value.device)
                new_bias[:out_dim] = value
                new_state_dict[key] = new_bias
            else:
                new_state_dict[key] = value
        else:
            # Other weights don't need modification
            new_state_dict[key] = value

    return new_state_dict


def load_gated_mmdit_from_original(
    original_state_dict: Dict[str, torch.Tensor],
    gate_type: str = "headwise",
    depth: int = 38,
) -> Dict[str, torch.Tensor]:
    """
    Load original MMDiT weights into a GatedMMDiT model.

    Args:
        original_state_dict: Original MMDiT state dict
        gate_type: "headwise" or "elementwise"
        depth: Number of layers (used to calculate num_heads)

    Returns:
        State dict compatible with GatedMMDiT
    """
    num_heads = depth  # In SD3, num_heads == depth
    return convert_attention_linears_to_gated(original_state_dict, gate_type, num_heads)


def collect_gate_statistics(mmdit, prefix: str = "") -> Dict[str, float]:
    """
    Collect gate statistics from all GatedMMDiTBlocks for tensorboard logging.

    Args:
        mmdit: The GatedMMDiT model
        prefix: Prefix for the metric names

    Returns:
        Dictionary of gate statistics suitable for tensorboard
    """
    all_stats = {}

    if not hasattr(mmdit, 'joint_blocks'):
        return all_stats

    gate_means = []
    gate_sparsities = []

    for block_idx, block in enumerate(mmdit.joint_blocks):
        if hasattr(block, 'get_gate_statistics'):
            block_stats = block.get_gate_statistics()

            for part_name, part_stats in block_stats.items():
                for attn_name, attn_stats in part_stats.items():
                    for stat_name, stat_value in attn_stats.items():
                        key = f"{prefix}block_{block_idx}/{part_name}/{attn_name}/{stat_name}"
                        all_stats[key] = stat_value

                        if stat_name == "gate_mean":
                            gate_means.append(stat_value)
                        elif stat_name == "gate_sparsity":
                            gate_sparsities.append(stat_value)

    # Add aggregated statistics
    if gate_means:
        all_stats[f"{prefix}gate_mean_overall"] = sum(gate_means) / len(gate_means)
    if gate_sparsities:
        all_stats[f"{prefix}gate_sparsity_overall"] = sum(gate_sparsities) / len(gate_sparsities)

    return all_stats


# Import additional dependencies from sd3_models
from .sd3_models import (
    PatchEmbed,
    UnPatch,
    TimestepEmbedding,
    Embedder,
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_torch,
    get_scaled_2d_sincos_pos_embed,
    get_bucketed_pos_embed,
    default,
    SD3Params,
)
from library import custom_offloading_utils
import numpy as np
import einops


class GatedMMDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone and Gated Attention.
    """

    # prepare pos_embed for latent size * 2
    POS_EMBED_MAX_RATIO = 1.5

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        depth: int = 28,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        adm_in_channels: Optional[int] = None,
        context_embedder_in_features: Optional[int] = None,
        context_embedder_out_features: Optional[int] = None,
        use_checkpoint: bool = False,
        register_length: int = 0,
        attn_mode: str = "torch",
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        out_channels: Optional[int] = None,
        pos_embed_scaling_factor: Optional[float] = None,
        pos_embed_offset: Optional[float] = None,
        pos_embed_max_size: Optional[int] = None,
        num_patches=None,
        qk_norm: Optional[str] = None,
        x_block_self_attn_layers: Optional[list[int]] = [],
        qkv_bias: bool = True,
        pos_emb_random_crop_rate: float = 0.0,
        use_scaled_pos_embed: bool = False,
        pos_embed_latent_sizes: Optional[list[int]] = None,
        use_bucketed_pos_embed: bool = False,
        model_type: str = "sd3m",
        gate_type: str = "headwise",  # "headwise" or "elementwise" or "none"
    ):
        super().__init__()
        self._model_type = model_type
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = default(out_channels, default_out_channels)
        self.patch_size = patch_size
        self.pos_embed_scaling_factor = pos_embed_scaling_factor
        self.pos_embed_offset = pos_embed_offset
        self.pos_embed_max_size = pos_embed_max_size
        self.x_block_self_attn_layers = x_block_self_attn_layers
        self.pos_emb_random_crop_rate = pos_emb_random_crop_rate
        self.use_bucketed_pos_embed = use_bucketed_pos_embed
        self.gradient_checkpointing = use_checkpoint
        self.gate_type = gate_type

        # apply magic --> this defines a head_size of 64
        self.hidden_size = 64 * depth
        num_heads = depth

        self.num_heads = num_heads
        self.depth = depth

        self.enable_scaled_pos_embed(use_scaled_pos_embed, pos_embed_latent_sizes)

        self.x_embedder = PatchEmbed(
            input_size,
            patch_size,
            in_channels,
            self.hidden_size,
            bias=True,
            strict_img_size=self.pos_embed_max_size is None,
        )
        self.t_embedder = TimestepEmbedding(self.hidden_size)

        self.y_embedder = None
        if adm_in_channels is not None:
            assert isinstance(adm_in_channels, int)
            self.y_embedder = Embedder(adm_in_channels, self.hidden_size)

        if context_embedder_in_features is not None:
            self.context_embedder = nn.Linear(context_embedder_in_features, context_embedder_out_features)
        else:
            self.context_embedder = nn.Identity()

        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, self.hidden_size))

        if num_patches is not None:
            self.register_buffer(
                "pos_embed",
                torch.empty(1, num_patches, self.hidden_size),
            )
        else:
            self.pos_embed = None

        self.use_checkpoint = use_checkpoint
        self.joint_blocks = nn.ModuleList(
            [
                GatedMMDiTBlock(
                    self.hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_mode=attn_mode,
                    qkv_bias=qkv_bias,
                    pre_only=i == depth - 1,
                    rmsnorm=rmsnorm,
                    scale_mod_only=scale_mod_only,
                    swiglu=swiglu,
                    qk_norm=qk_norm,
                    x_block_self_attn=(i in self.x_block_self_attn_layers),
                    gate_type=gate_type,
                )
                for i in range(depth)
            ]
        )
        for block in self.joint_blocks:
            block.gradient_checkpointing = use_checkpoint

        self.final_layer = UnPatch(self.hidden_size, patch_size, self.out_channels)

        self.blocks_to_swap = None
        self.offloader = None
        self.num_blocks = len(self.joint_blocks)

    def enable_scaled_pos_embed(self, use_scaled_pos_embed: bool, latent_sizes: Optional[list[int]]):
        self.use_scaled_pos_embed = use_scaled_pos_embed

        if self.use_scaled_pos_embed:
            self.pos_embed = self.pos_embed.cpu() if self.pos_embed is not None else None

            latent_sizes = list(set(latent_sizes)) if latent_sizes else []
            latent_sizes = sorted(latent_sizes)

            patched_sizes = [latent_size // self.patch_size for latent_size in latent_sizes]

            max_areas = []
            for i in range(1, len(patched_sizes)):
                prev_area = patched_sizes[i - 1] ** 2
                area = patched_sizes[i] ** 2
                max_areas.append((prev_area + area) // 2)

            max_areas.append(int((patched_sizes[-1] * GatedMMDiT.POS_EMBED_MAX_RATIO) ** 2)) if patched_sizes else None

            self.resolution_area_to_latent_size = [(area, latent_size) for area, latent_size in zip(max_areas, patched_sizes)] if max_areas else []

            self.resolution_pos_embeds = {}
            for patched_size in patched_sizes:
                grid_size = int(patched_size * GatedMMDiT.POS_EMBED_MAX_RATIO)
                pos_embed = get_scaled_2d_sincos_pos_embed(self.hidden_size, grid_size, sample_size=patched_size)
                pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
                self.resolution_pos_embeds[patched_size] = pos_embed
        else:
            self.resolution_area_to_latent_size = None
            self.resolution_pos_embeds = None

    @property
    def model_type(self):
        return self._model_type

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        for block in self.joint_blocks:
            block.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        for block in self.joint_blocks:
            block.gradient_checkpointing = False

    def set_pos_emb_random_crop_rate(self, rate: float):
        self.pos_emb_random_crop_rate = rate

    def cropped_pos_embed(self, h, w, device=None, random_crop: bool = False):
        p = self.x_embedder.patch_size
        h = (h + 1) // p
        w = (w + 1) // p
        if self.pos_embed is None:
            return get_2d_sincos_pos_embed_torch(self.hidden_size, w, h, device=device)
        assert self.pos_embed_max_size is not None
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)

        if not random_crop:
            top = (self.pos_embed_max_size - h) // 2
            left = (self.pos_embed_max_size - w) // 2
        else:
            top = torch.randint(0, self.pos_embed_max_size - h + 1, (1,)).item()
            left = torch.randint(0, self.pos_embed_max_size - w + 1, (1,)).item()

        spatial_pos_embed = self.pos_embed.reshape(
            1,
            self.pos_embed_max_size,
            self.pos_embed_max_size,
            self.pos_embed.shape[-1],
        )
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    def cropped_scaled_pos_embed(self, h, w, device=None, dtype=None, random_crop: bool = False):
        p = self.x_embedder.patch_size
        h = (h + 1) // p
        w = (w + 1) // p

        area = h * w
        patched_size = None
        for area_, patched_size_ in self.resolution_area_to_latent_size:
            if area <= area_:
                patched_size = patched_size_
                break
        if patched_size is None:
            patched_size = self.resolution_area_to_latent_size[-1][1] if self.resolution_area_to_latent_size else h

        pos_embed = self.resolution_pos_embeds.get(patched_size)
        if pos_embed is None:
            patched_size = max(h, w)
            grid_size = int(patched_size * GatedMMDiT.POS_EMBED_MAX_RATIO)
            pos_embed = get_scaled_2d_sincos_pos_embed(self.hidden_size, grid_size, sample_size=patched_size)
            pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
            self.resolution_pos_embeds[patched_size] = pos_embed

        pos_embed_size = round(math.sqrt(pos_embed.shape[1]))

        if not random_crop:
            top = (pos_embed_size - h) // 2
            left = (pos_embed_size - w) // 2
        else:
            top = torch.randint(0, pos_embed_size - h + 1, (1,)).item()
            left = torch.randint(0, pos_embed_size - w + 1, (1,)).item()

        if pos_embed.device != device:
            pos_embed = pos_embed.to(device)
            self.resolution_pos_embeds[patched_size] = pos_embed
        if pos_embed.dtype != dtype:
            pos_embed = pos_embed.to(dtype)
            self.resolution_pos_embeds[patched_size] = pos_embed

        spatial_pos_embed = pos_embed.reshape(1, pos_embed_size, pos_embed_size, pos_embed.shape[-1])
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    def bucketed_pos_embed(self, h, w, device=None, dtype=None):
        p = self.x_embedder.patch_size
        h_latent = (h + 1) // p
        w_latent = (w + 1) // p

        target_resolution_pixels = 1440
        s_latent = target_resolution_pixels // p

        max_aspect_ratio = 8.0

        h_max = int(s_latent * max_aspect_ratio**0.5)
        w_max = int(s_latent * max_aspect_ratio**0.5)

        h_max = max(h_max, h_latent)
        w_max = max(w_max, w_latent)

        return get_bucketed_pos_embed(
            self.hidden_size,
            h_max,
            w_max,
            s_latent,
            h_latent,
            w_latent,
            device=device,
            dtype=dtype
        )

    def enable_block_swap(self, num_blocks: int, device: torch.device):
        self.blocks_to_swap = num_blocks

        assert (
            self.blocks_to_swap <= self.num_blocks - 2
        ), f"Cannot swap more than {self.num_blocks - 2} blocks. Requested: {self.blocks_to_swap} blocks."

        self.offloader = custom_offloading_utils.ModelOffloader(
            self.joint_blocks, self.num_blocks, self.blocks_to_swap, device
        )
        print(f"GatedSD3: Block swap enabled. Swapping {num_blocks} blocks, total blocks: {self.num_blocks}, device: {device}.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            save_blocks = self.joint_blocks
            self.joint_blocks = None

        self.to(device)

        if self.blocks_to_swap:
            self.joint_blocks = save_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.joint_blocks)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of GatedDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, D) tensor of class labels
        """
        pos_emb_random_crop = (
            False if self.pos_emb_random_crop_rate == 0.0 else torch.rand(1).item() < self.pos_emb_random_crop_rate
        )

        B, C, H, W = x.shape

        if self.use_bucketed_pos_embed:
            pos_embed = self.bucketed_pos_embed(H, W, device=x.device, dtype=x.dtype)
        elif not self.use_scaled_pos_embed:
            pos_embed = self.cropped_pos_embed(H, W, device=x.device, random_crop=pos_emb_random_crop).to(dtype=x.dtype)
        else:
            pos_embed = self.cropped_scaled_pos_embed(H, W, device=x.device, dtype=x.dtype, random_crop=pos_emb_random_crop)
        x = self.x_embedder(x) + pos_embed
        del pos_embed

        c = self.t_embedder(t, dtype=x.dtype)
        if y is not None and self.y_embedder is not None:
            y = self.y_embedder(y)
            c = c + y

        if context is not None:
            context = self.context_embedder(context)

        if self.register_length > 0:
            context = torch.cat(
                (einops.repeat(self.register, "1 ... -> b ...", b=x.shape[0]), default(context, torch.Tensor([]).type_as(x))), 1
            )

        if not self.blocks_to_swap:
            for block in self.joint_blocks:
                context, x = block(context, x, c)
        else:
            for block_idx, block in enumerate(self.joint_blocks):
                self.offloader.wait_for_block(block_idx)
                context, x = block(context, x, c)
                self.offloader.submit_move_blocks(self.joint_blocks, block_idx)

        x = self.final_layer(x, c, H, W)
        return x[:, :, :H, :W]

    def set_gate_layers(self, layer_ids: Optional[List[int]] = None):
        """Enable gated attention only for specified layers. Other layers bypass gating.

        Args:
            layer_ids: List of 0-based layer indices to enable gating. None means keep all enabled.
        """
        if layer_ids is None:
            return
        enabled = set(layer_ids)
        for idx, block in enumerate(self.joint_blocks):
            if hasattr(block, "set_gate_enabled"):
                block.set_gate_enabled(idx in enabled)
        logger.info(f"Gate layers set: {sorted(enabled)} out of {len(self.joint_blocks)} blocks")

    def set_log_gate_stats(self, enabled: bool):
        """Enable or disable gate statistics logging for all blocks."""
        for block in self.joint_blocks:
            if hasattr(block, 'set_log_gate_stats'):
                block.set_log_gate_stats(enabled)

    def get_gate_statistics(self) -> Dict[str, float]:
        """Get aggregated gate statistics for tensorboard logging."""
        return collect_gate_statistics(self, prefix="")


def create_gated_sd3_mmdit(params: SD3Params, attn_mode: str = "torch", gate_type: str = "headwise") -> GatedMMDiT:
    """Create a GatedMMDiT model from SD3Params."""
    mmdit = GatedMMDiT(
        input_size=None,
        pos_embed_max_size=params.pos_embed_max_size,
        patch_size=params.patch_size,
        in_channels=16,
        adm_in_channels=params.adm_in_channels,
        context_embedder_in_features=params.context_embedder_in_features,
        context_embedder_out_features=params.context_embedder_out_features,
        depth=params.depth,
        mlp_ratio=4,
        qk_norm=params.qk_norm,
        x_block_self_attn_layers=params.x_block_self_attn_layers,
        num_patches=params.num_patches,
        attn_mode=attn_mode,
        model_type=params.model_type,
        use_bucketed_pos_embed=params.use_bucketed_pos_embed,
        gate_type=gate_type,
    )
    return mmdit


def load_gated_mmdit(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype] = None,
    device: str = "cpu",
    gate_type: str = "headwise",
) -> GatedMMDiT:
    """
    Load a GatedMMDiT model from an original MMDiT state dict.

    This function:
    1. Detects the model type from state dict
    2. Creates a GatedMMDiT model
    3. Converts the original weights to gated format
    4. Loads the converted weights

    Args:
        state_dict: Original MMDiT state dict (with model.diffusion_model. prefix removed)
        dtype: Target dtype
        device: Target device
        gate_type: "headwise" or "elementwise"

    Returns:
        GatedMMDiT model with loaded weights
    """
    from .sd3_utils import detect_sd3_model_type

    # Detect model type
    params = detect_sd3_model_type(state_dict)

    # Create gated model
    mmdit = create_gated_sd3_mmdit(params, attn_mode="torch", gate_type=gate_type)

    # Convert original weights to gated format
    gated_state_dict = load_gated_mmdit_from_original(state_dict, gate_type=gate_type, depth=params.depth)

    # Load weights
    info = mmdit.load_state_dict(gated_state_dict, strict=False)
    if info.missing_keys:
        logger.info(f"Missing keys when loading GatedMMDiT: {info.missing_keys}")
    if info.unexpected_keys:
        logger.warning(f"Unexpected keys when loading GatedMMDiT: {info.unexpected_keys}")

    # Move to device and dtype
    if dtype is not None:
        mmdit.to(dtype)
    mmdit.to(device)

    logger.info(f"Loaded GatedMMDiT with gate_type={gate_type}, depth={params.depth}")

    return mmdit
