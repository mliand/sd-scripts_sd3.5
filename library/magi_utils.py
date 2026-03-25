import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _str_to_torch_dtype(dtype: Optional[str], default: torch.dtype = torch.bfloat16) -> torch.dtype:
    if dtype is None:
        return default
    value = dtype.strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[value]


def ensure_magi_compiler_stub() -> None:
    if importlib.util.find_spec("magi_compiler") is not None:
        return

    logger.warning("magi_compiler not found, installing runtime no-op stub.")

    magi_compiler_mod = types.ModuleType("magi_compiler")

    def magi_compile(*args, **kwargs):
        def decorator(obj):
            return obj

        return decorator

    magi_compiler_mod.magi_compile = magi_compile

    api_mod = types.ModuleType("magi_compiler.api")

    def magi_register_custom_op(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    api_mod.magi_register_custom_op = magi_register_custom_op

    config_mod = types.ModuleType("magi_compiler.config")

    @dataclass
    class _OffloadConfig:
        gpu_resident_weight_ratio: float = 1.0

    @dataclass
    class CompileConfig:
        offload_config: _OffloadConfig = field(default_factory=_OffloadConfig)

    config_mod.CompileConfig = CompileConfig

    sys.modules["magi_compiler"] = magi_compiler_mod
    sys.modules["magi_compiler.api"] = api_mod
    sys.modules["magi_compiler.config"] = config_mod

def patch_single_process_parallel_state() -> None:
    """
    daVinci's Ulysses scheduler expects cp_world_size>=1 and a valid cp_group.
    In non-distributed training, original helpers may return cp_world_size=0 and raise on cp_group.
    This patch makes single-process behavior equivalent to cp=1.
    """

    from inference.infra.distributed import parallel_state
    import inference.infra.distributed as distributed_api
    import inference.infra.parallelism.ulysses_scheduler as ulysses_mod

    def _safe_get_cp_world_size() -> int:
        ws = parallel_state.get_cp_world_size()
        return 1 if ws <= 0 else ws

    def _safe_get_cp_group(*args, **kwargs):
        try:
            return parallel_state.get_cp_group(check_initialized=False)
        except Exception:
            return None

    distributed_api.get_cp_world_size = _safe_get_cp_world_size
    distributed_api.get_cp_group = _safe_get_cp_group
    ulysses_mod.get_cp_world_size = _safe_get_cp_world_size
    ulysses_mod.get_cp_group = _safe_get_cp_group


def import_magi_components():
    ensure_magi_compiler_stub()

    from inference.common.config import DataProxyConfig, ModelConfig
    from inference.model.dit.dit_module import DiTModel
    from inference.pipeline.data_proxy import MagiDataProxy
    from inference.model.vae2_2.vae2_2_model import get_vae2_2
    from inference.pipeline.prompt_process import get_padded_t5_gemma_embedding

    patch_single_process_parallel_state()

    return {
        "ModelConfig": ModelConfig,
        "DataProxyConfig": DataProxyConfig,
        "DiTModel": DiTModel,
        "MagiDataProxy": MagiDataProxy,
        "get_vae2_2": get_vae2_2,
        "get_padded_t5_gemma_embedding": get_padded_t5_gemma_embedding,
    }


def load_magi_configs(config_load_path: Optional[str] = None):
    comps = import_magi_components()
    ModelConfig = comps["ModelConfig"]
    DataProxyConfig = comps["DataProxyConfig"]

    model_cfg = ModelConfig()
    data_proxy_cfg = DataProxyConfig()
    t5_target_length = 640

    if config_load_path is not None:
        with open(config_load_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        if "arch_config" in cfg and isinstance(cfg["arch_config"], dict):
            model_cfg = ModelConfig(**cfg["arch_config"])
        if "evaluation_config" in cfg and isinstance(cfg["evaluation_config"], dict):
            eval_cfg = cfg["evaluation_config"]
            if "data_proxy_config" in eval_cfg and isinstance(eval_cfg["data_proxy_config"], dict):
                data_proxy_cfg = DataProxyConfig(**eval_cfg["data_proxy_config"])
            t5_target_length = int(eval_cfg.get("t5_gemma_target_length", t5_target_length))

    # Post-processing that parse_config() normally does.
    model_cfg.num_heads_q = model_cfg.hidden_size // model_cfg.head_dim
    model_cfg.num_heads_kv = model_cfg.num_query_groups

    return model_cfg, data_proxy_cfg, t5_target_length


def _load_magi_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    from inference.infra.checkpoint.load_model_checkpoint import load_sharded_safetensors_parallel_with_progress

    if os.path.isdir(ckpt_path):
        return load_sharded_safetensors_parallel_with_progress(ckpt_path)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    return load_file(ckpt_path)


def load_magi_dit_model(
    checkpoint_path: str,
    model_config,
    device: torch.device,
    model_dtype: torch.dtype,
):
    comps = import_magi_components()
    DiTModel = comps["DiTModel"]

    logger.info(f"Building DiT model from daVinci config: hidden={model_config.hidden_size}, layers={model_config.num_layers}")
    model = DiTModel(model_config)

    logger.info(f"Loading DiT checkpoint from {checkpoint_path}")
    sd = _load_magi_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"Missing keys while loading DiT checkpoint: {len(missing)}")
    if unexpected:
        logger.warning(f"Unexpected keys while loading DiT checkpoint: {len(unexpected)}")

    model = model.to(device=device)
    model = model.to(dtype=model_dtype)
    return model


@dataclass
class MagiTrainInput:
    x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: Sequence[int]
    txt_feat: torch.Tensor
    txt_feat_len: Sequence[int]


class MagiModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, data_proxy: Any, audio_in_channels: int):
        super().__init__()
        self.model = model
        self.data_proxy = data_proxy
        self.audio_in_channels = audio_in_channels

    def forward(self, noisy_video: torch.Tensor, text_embeds: torch.Tensor, text_lengths: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = noisy_video.shape[0]
        audio_x_t = torch.zeros(
            (bsz, 0, self.audio_in_channels),
            device=noisy_video.device,
            dtype=noisy_video.dtype,
        )
        audio_feat_len = [0] * bsz

        packed = MagiTrainInput(
            x_t=noisy_video,
            audio_x_t=audio_x_t,
            audio_feat_len=audio_feat_len,
            txt_feat=text_embeds,
            txt_feat_len=list(text_lengths),
        )

        model_inputs = self.data_proxy.process_input(packed)
        pred_tokens = self.model(*model_inputs)
        pred_video, pred_audio = self.data_proxy.process_output(pred_tokens)
        return pred_video, pred_audio


def load_jsonl_records(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {line_no} is not an object: {line}")
            records.append(obj)
    return records


def save_jsonl_records(path: str, records: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def resolve_data_path(base_manifest_path: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    base_dir = os.path.dirname(os.path.abspath(base_manifest_path))
    return os.path.abspath(os.path.join(base_dir, value))


def build_item_key(record: dict, index: int) -> str:
    if "id" in record and record["id"]:
        return str(record["id"])

    video = record.get("video") or record.get("video_path") or ""
    if video:
        stem = Path(video).stem
        if stem:
            return f"{stem}_{index:06d}"

    return f"item_{index:06d}"


def get_caption(record: dict) -> str:
    for key in ("caption", "prompt", "text"):
        value = record.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    raise ValueError(f"Record missing caption/prompt/text: {record}")


def default_latent_cache_name(item_key: str, frame_count: int, height: int, width: int) -> str:
    return f"{item_key}_{frame_count:03d}_{height:04d}x{width:04d}_magi_latent.safetensors"


def default_te_cache_name(item_key: str) -> str:
    return f"{item_key}_magi_te.safetensors"


__all__ = [
    "_str_to_torch_dtype",
    "MagiModelWrapper",
    "build_item_key",
    "default_latent_cache_name",
    "default_te_cache_name",
    "get_caption",
    "import_magi_components",
    "load_jsonl_records",
    "load_magi_configs",
    "load_magi_dit_model",
    "resolve_data_path",
    "save_jsonl_records",
]
