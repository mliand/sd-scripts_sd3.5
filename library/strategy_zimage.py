import os
from typing import Any, List, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer

from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class ZImageTokenizeStrategy(TokenizeStrategy):
    def __init__(
        self,
        tokenizer_id_or_path: str,
        max_length: int = 512,
        tokenizer_cache_dir: Optional[str] = None,
        apply_chat_template: bool = True,
    ) -> None:
        self.max_length = max_length
        self.apply_chat_template = apply_chat_template

        cache_dir = tokenizer_cache_dir
        logger.info(f"Loading Z-Image tokenizer from {tokenizer_id_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id_or_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format_prompt(self, prompt: str) -> str:
        if not self.apply_chat_template or not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt

        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text_list = [text] if isinstance(text, str) else text
        formatted = [self._format_prompt(p) for p in text_list]

        tokens = self.tokenizer(
            formatted,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return [tokens.input_ids, tokens.attention_mask]


class ZImageTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self) -> None:
        pass

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        text_encoder = models[0] if len(models) > 0 else None
        if text_encoder is None:
            return [None, None]

        input_ids, attention_mask = tokens
        input_ids = input_ids.to(text_encoder.device)
        attention_mask = attention_mask.to(text_encoder.device).bool()

        outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        prompt_embeds = outputs.hidden_states[-2]
        return [prompt_embeds, attention_mask]


class ZImageTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    ZIMAGE_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_zimage_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        max_length: int = 512,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)
        self.max_length = max_length

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + ZImageTextEncoderOutputsCachingStrategy.ZIMAGE_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            if "prompt_embeds" not in npz:
                return False
            if "attention_mask" not in npz:
                return False
            if "max_length" not in npz:
                return False
            if int(npz["max_length"]) != self.max_length:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        prompt_embeds = data["prompt_embeds"]
        attention_mask = data["attention_mask"]
        return [prompt_embeds, attention_mask]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        captions = [info.caption for info in infos]
        tokens_and_masks = tokenize_strategy.tokenize(captions)

        with torch.no_grad():
            prompt_embeds, attention_mask = text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens_and_masks)

        if prompt_embeds.dtype == torch.bfloat16:
            prompt_embeds = prompt_embeds.float()
        prompt_embeds = prompt_embeds.cpu().numpy()
        attention_mask = attention_mask.cpu().numpy()

        for i, info in enumerate(infos):
            prompt_embeds_i = prompt_embeds[i]
            attention_mask_i = attention_mask[i]

            if self.cache_to_disk:
                np.savez(
                    info.text_encoder_outputs_npz,
                    prompt_embeds=prompt_embeds_i,
                    attention_mask=attention_mask_i,
                    max_length=self.max_length,
                )
            else:
                info.text_encoder_outputs = (prompt_embeds_i, attention_mask_i)


class ZImageLatentsCachingStrategy(LatentsCachingStrategy):
    ZIMAGE_LATENTS_NPZ_SUFFIX = "_zimage.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return ZImageLatentsCachingStrategy.ZIMAGE_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: tuple[int, int]) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + ZImageLatentsCachingStrategy.ZIMAGE_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(self, bucket_reso: tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool) -> bool:
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: tuple[int, int]
    ) -> tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)  # support multi-resolution

    def cache_batch_latents(self, model: Any, batch: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        def encode_by_vae(img_tensor: torch.Tensor) -> torch.Tensor:
            latents = model.encode(img_tensor).latent_dist.mode()
            return latents

        self._default_cache_batch_latents(
            encode_by_vae, model.device, model.dtype, batch, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )
