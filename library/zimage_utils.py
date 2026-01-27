import os
from typing import Optional, Tuple, Union

import torch
from library.zimage_autoencoder import AutoencoderKL, load_autoencoder_kl
from library.zimage_model import load_zimage_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


DEFAULT_MAX_SEQUENCE_LENGTH = 512


def load_tokenizer(tokenizer_id_or_path: str, tokenizer_cache_dir: Optional[str] = None):
    logger.info(f"Loading tokenizer from {tokenizer_id_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id_or_path,
        cache_dir=tokenizer_cache_dir,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_text_encoder(
    text_encoder_id_or_path: str,
    dtype: torch.dtype,
    device: Union[str, torch.device],
):
    logger.info(f"Loading text encoder from {text_encoder_id_or_path}")
    text_encoder = AutoModelForCausalLM.from_pretrained(
        text_encoder_id_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    text_encoder.to(device)
    text_encoder.eval()
    return text_encoder


def load_transformer(
    path: str,
    dtype: torch.dtype,
    device: Union[str, torch.device],
):
    logger.info(f"Loading Z-Image transformer from {path}")
    transformer = load_zimage_model(path, dtype=dtype, device=device)
    return transformer


def load_vae(
    path: str,
    dtype: torch.dtype,
    device: Union[str, torch.device],
):
    logger.info(f"Loading Z-Image VAE from {path}")
    if os.path.isdir(path):
        found = False
        for candidate in (os.path.join(path, "vae"), path):
            if not os.path.isdir(candidate):
                continue
            files = [f for f in os.listdir(candidate) if f.endswith(".safetensors")]
            if files:
                path = os.path.join(candidate, sorted(files)[0])
                found = True
                break
        if not found:
            raise FileNotFoundError(f"No .safetensors found under {path}")
    vae = load_autoencoder_kl(path, device="cpu")
    vae.to(device)
    vae.to(torch.float32)
    vae.eval()
    return vae


def scale_shift_latents(latents: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    scale = vae.config.scaling_factor
    shift = vae.config.shift_factor
    return (latents - shift) * scale
