from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)
LORA_PREFIX_MAGI = "lora_unet"


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_module: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        multiplier: float = 1.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        if not hasattr(base_module, "in_features") or not hasattr(base_module, "out_features"):
            raise TypeError(f"Unsupported module for LoRA: {type(base_module)}")

        in_features = int(getattr(base_module, "in_features"))
        out_features = int(getattr(base_module, "out_features"))
        if in_features <= 0 or out_features <= 0:
            raise ValueError(f"Invalid in/out features for LoRA: {in_features}, {out_features}")

        self.org_module_ref = [base_module]
        for p in base_module.parameters():
            p.requires_grad = False

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)
        self.multiplier = float(multiplier)
        self.lora_down = nn.Linear(in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, out_features, bias=False)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        base_out = self.org_module_ref[0](x, *args, **kwargs)
        lora_x = self.dropout(x).to(self.lora_down.weight.dtype)
        lora_out = self.lora_up(self.lora_down(lora_x)) * self.scale * self.multiplier
        return base_out + lora_out.to(base_out.dtype)


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue

        if hasattr(module, "lora_name"):
            lora_name = str(module.lora_name)
        elif module_name.startswith(f"{LORA_PREFIX_MAGI}_"):
            lora_name = module_name
        else:
            # Compatibility path when iterating over the wrapped backbone model.
            lora_name = f"{LORA_PREFIX_MAGI}_{module_name.replace('.', '_')}"

        sd[f"{lora_name}.lora_down.weight"] = module.lora_down.weight.detach().cpu().contiguous()
        sd[f"{lora_name}.lora_up.weight"] = module.lora_up.weight.detach().cpu().contiguous()
        sd[f"{lora_name}.alpha"] = torch.tensor([module.alpha], dtype=torch.float32)
        sd[f"{lora_name}.rank"] = torch.tensor([module.rank], dtype=torch.int32)
    return sd


def _str_to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _set_child_module(parent: torch.nn.Module, child_name: str, module: torch.nn.Module) -> None:
    if isinstance(parent, (torch.nn.ModuleList, torch.nn.Sequential)):
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def _load_weights_file(file: str):
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file

        return load_file(file)
    return torch.load(file, map_location="cpu")


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoder,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = float(network_dim)

    target_modules = kwargs.get("target_modules", "linear_qkv,linear_proj,up_gate_proj,down_proj")
    target_modules = [x.strip() for x in str(target_modules).split(",") if x.strip()]
    train_adapter = _str_to_bool(kwargs.get("train_adapter", False))

    rank_dropout = kwargs.get("rank_dropout", None)
    module_dropout = kwargs.get("module_dropout", None)
    if rank_dropout is not None:
        logger.warning("rank_dropout is not supported by lora_magi; ignoring.")
    if module_dropout is not None:
        logger.warning("module_dropout is not supported by lora_magi; ignoring.")

    network = LoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=int(network_dim),
        alpha=float(network_alpha),
        dropout=float(neuron_dropout) if neuron_dropout is not None else 0.0,
        target_modules=target_modules,
        train_adapter=train_adapter,
    )
    return network


def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        weights_sd = _load_weights_file(file)

    modules_dim: Dict[str, int] = {}
    modules_alpha: Dict[str, float] = {}
    for key, value in weights_sd.items():
        if key.endswith(".lora_down.weight"):
            lora_name = key[: -len(".lora_down.weight")]
            modules_dim[lora_name] = int(value.shape[0])
        elif key.endswith(".alpha"):
            lora_name = key[: -len(".alpha")]
            modules_alpha[lora_name] = float(value.reshape(-1)[0].item())

    target_modules = kwargs.get("target_modules", "linear_qkv,linear_proj,up_gate_proj,down_proj")
    target_modules = [x.strip() for x in str(target_modules).split(",") if x.strip()]
    train_adapter = _str_to_bool(kwargs.get("train_adapter", False))

    network = LoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        target_modules=target_modules,
        train_adapter=train_adapter,
    )
    return network, weights_sd


class LoRANetwork(torch.nn.Module):
    LORA_PREFIX_MAGI = "lora_unet"

    def __init__(
        self,
        text_encoders,
        mmdit: torch.nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, float]] = None,
        target_modules: Optional[Sequence[str]] = None,
        train_adapter: bool = False,
    ) -> None:
        super().__init__()
        self.multiplier = float(multiplier)
        self.lora_dim = int(lora_dim)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.target_modules = list(target_modules or ["linear_qkv", "linear_proj", "up_gate_proj", "down_proj"])
        self.train_adapter = bool(train_adapter)
        self.modules_dim = dict(modules_dim) if modules_dim is not None else None
        self.modules_alpha = dict(modules_alpha) if modules_alpha is not None else {}

        self.text_encoder_loras: List[LoRALinear] = []
        self.unet_loras: List[LoRALinear] = []
        self._module_specs: List[Tuple[str, str, int, float]] = []
        self._applied = False

        self._collect_module_specs(mmdit)

    @staticmethod
    def _is_supported_module(module: torch.nn.Module) -> bool:
        if isinstance(module, LoRALinear):
            return False
        return hasattr(module, "in_features") and hasattr(module, "out_features") and hasattr(module, "weight")

    @classmethod
    def _to_lora_name(cls, module_name: str) -> str:
        return cls.LORA_PREFIX_MAGI + "_" + module_name.replace(".", "_")

    def _collect_module_specs(self, mmdit: torch.nn.Module):
        seen = set()
        for module_name, module in mmdit.named_modules():
            if module_name == "":
                continue
            if not self._is_supported_module(module):
                continue
            if not self.train_adapter and module_name.startswith("adapter."):
                continue

            lora_name = self._to_lora_name(module_name)

            if self.modules_dim is not None:
                dim = self.modules_dim.get(lora_name, None)
                if dim is None:
                    # Backward compatibility for early Magi checkpoints using raw module_name keys.
                    dim = self.modules_dim.get(module_name, None)
                    if dim is None:
                        continue
                    alpha = self.modules_alpha.get(module_name, float(dim))
                else:
                    alpha = self.modules_alpha.get(lora_name, float(dim))
            else:
                if not any(t in module_name for t in self.target_modules):
                    continue
                dim = self.lora_dim
                alpha = self.alpha

            if dim <= 0:
                continue
            if lora_name in seen:
                continue

            self._module_specs.append((module_name, lora_name, int(dim), float(alpha)))
            seen.add(lora_name)

        logger.info(f"create LoRA for Magi DiT: {len(self._module_specs)} modules")
        if len(self._module_specs) == 0:
            logger.warning("No target modules found for lora_magi.")

    def _inject_module(self, mmdit: torch.nn.Module, module_name: str, lora_name: str, dim: int, alpha: float):
        if "." in module_name:
            parent_name, child_name = module_name.rsplit(".", 1)
            parent = mmdit.get_submodule(parent_name)
        else:
            parent = mmdit
            child_name = module_name

        current = parent[int(child_name)] if isinstance(parent, (torch.nn.ModuleList, torch.nn.Sequential)) else getattr(parent, child_name)
        if isinstance(current, LoRALinear):
            wrapper = current
            wrapper.multiplier = self.multiplier
        else:
            wrapper = LoRALinear(
                current,
                rank=dim,
                alpha=alpha,
                dropout=self.dropout,
                multiplier=self.multiplier,
            )
            _set_child_module(parent, child_name, wrapper)
        wrapper.lora_name = lora_name

        self.add_module(lora_name, wrapper)
        self.unet_loras.append(wrapper)

    def set_multiplier(self, multiplier):
        self.multiplier = float(multiplier)
        for lora in self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled: bool):
        self.set_multiplier(1.0 if is_enabled else 0.0)

    def apply_to(self, text_encoders, mmdit, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info("lora_magi does not modify text encoder modules; ignoring text encoder target.")
        self.text_encoder_loras = []

        if not apply_unet:
            self.unet_loras = []
            return

        if self._applied:
            return

        for module_name, lora_name, dim, alpha in self._module_specs:
            self._inject_module(mmdit, module_name, lora_name, dim, alpha)

        self._applied = True
        logger.info(f"enable LoRA for Magi DiT: {len(self.unet_loras)} modules")

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoders, mmdit, weights_sd, dtype=None, device=None):
        if not self._applied:
            self.apply_to(text_encoders, mmdit, apply_text_encoder=False, apply_unet=True)
        self.load_state_dict(weights_sd, strict=False)

        for lora in self.unet_loras:
            org = lora.org_module_ref[0]
            if not hasattr(org, "weight"):
                continue
            down = lora.lora_down.weight.to(org.weight.device, dtype=org.weight.dtype)
            up = lora.lora_up.weight.to(org.weight.device, dtype=org.weight.dtype)
            delta = (up @ down) * lora.scale * lora.multiplier
            if delta.shape == org.weight.shape:
                org.weight.data.add_(delta)
        self.set_enabled(False)
        logger.info("weights are merged")

    def prepare_network(self, args):
        pass

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr=None):
        self.requires_grad_(True)
        lr = unet_lr if unet_lr is not None else default_lr
        params = list(self.parameters())
        if lr is None:
            return [{"params": params}]
        return [{"params": params, "lr": lr}]

    def enable_gradient_checkpointing(self):
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def load_state_dict(self, state_dict, strict=True):
        filtered = {
            k: v
            for k, v in state_dict.items()
            if k.endswith(".lora_down.weight") or k.endswith(".lora_up.weight")
        }
        info = super().load_state_dict(filtered, strict=False)

        for key, value in state_dict.items():
            if not key.endswith(".alpha"):
                continue
            lora_name = key[: -len(".alpha")]
            if not hasattr(self, lora_name):
                continue
            module = getattr(self, lora_name)
            if not isinstance(module, LoRALinear):
                continue
            module.alpha = float(value.reshape(-1)[0].item())
            module.scale = module.alpha / float(module.rank)
        return info

    def load_weights(self, file):
        weights_sd = _load_weights_file(file)
        info = self.load_state_dict(weights_sd, False)
        return info

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return get_lora_state_dict(self)

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                if state_dict[key].is_floating_point():
                    state_dict[key] = state_dict[key].detach().clone().to("cpu").to(dtype)
                else:
                    state_dict[key] = state_dict[key].detach().clone().to("cpu")

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
