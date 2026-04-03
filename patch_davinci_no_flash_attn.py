"""
Patch daVinci DiT to use PyTorch SDPA instead of FlashAttention.
Run this before training if flash_attn is not available.
"""
import sys
import os

# Add inference module to path
inference_path = os.path.join(os.path.dirname(__file__), "inference")
if os.path.isdir(inference_path):
    sys.path.insert(0, inference_path)

try:
    from model.dit import dit_module
except ImportError:
    print("ERROR: Cannot import daVinci dit_module. Make sure inference/ directory exists.")
    sys.exit(1)

import torch
import torch.nn.functional as F

def patched_flash_attn_func(q, k, v):
    """
    Fallback to PyTorch SDPA when flash_attn is not available.

    Args:
        q, k, v: [batch, seqlen, num_heads, head_dim]

    Returns:
        attn_out: [batch, seqlen, num_heads, head_dim]
    """
    # Transpose to [batch, num_heads, seqlen, head_dim] for SDPA
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Use PyTorch scaled_dot_product_attention
    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

    # Transpose back to [batch, seqlen, num_heads, head_dim]
    attn_out = attn_out.transpose(1, 2)

    return attn_out

# Monkey patch the flash_attn_func
dit_module.flash_attn_func = patched_flash_attn_func

print("✓ Patched daVinci DiT to use PyTorch SDPA instead of FlashAttention")
