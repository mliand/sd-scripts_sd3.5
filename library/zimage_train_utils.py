import argparse
from typing import Optional

import torch
from safetensors.torch import save_file

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def save_models(
    ckpt_path: str,
    transformer: torch.nn.Module,
    text_encoder: Optional[torch.nn.Module],
    save_dtype: Optional[torch.dtype] = None,
):
    state_dict = {}
    for key, value in transformer.state_dict().items():
        if save_dtype is not None and value.dtype != save_dtype:
            value = value.detach().clone().to("cpu").to(save_dtype)
        state_dict[key] = value

    save_file(state_dict, ckpt_path)

    if text_encoder is not None:
        text_encoder_path = ckpt_path.replace(".safetensors", "_text_encoder.safetensors")
        te_state_dict = text_encoder.state_dict()
        save_file(te_state_dict, text_encoder_path)


def save_zimage_model_on_train_end(
    args: argparse.Namespace,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    transformer: torch.nn.Module,
    text_encoder: Optional[torch.nn.Module],
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        save_models(ckpt_file, transformer, text_encoder, save_dtype)

    from library import train_util

    train_util.save_sd_model_on_train_end_common(args, True, True, epoch, global_step, sd_saver, None)


def save_zimage_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    transformer: torch.nn.Module,
    text_encoder: Optional[torch.nn.Module],
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        save_models(ckpt_file, transformer, text_encoder, save_dtype)

    from library import train_util

    train_util.save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        True,
        True,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        None,
    )


def add_zimage_train_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--vae",
        type=str,
        required=True,
        help="path to Z-Image VAE (diffusers directory or safetensors file)",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="path or HF id for the Qwen text encoder",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="tokenizer path or HF id (defaults to --text_encoder)",
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=512,
        help="maximum token length for Qwen tokenizer",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="enable training of the Qwen text encoder",
    )
    parser.add_argument(
        "--disable_chat_template",
        action="store_true",
        help="do not apply tokenizer chat template for prompts",
    )
    parser.add_argument(
        "--timestep_sampling",
        type=str,
        default="shift",
        choices=["uniform", "sigmoid", "shift"],
        help="timestep sampling method for Z-Image training",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=3.0,
        help="discrete flow shift for shift-based timestep sampling",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help="sigmoid scale for timestep sampling (used by sigmoid/shift)",
    )
