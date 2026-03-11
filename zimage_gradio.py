import argparse
import os
import random
import threading
from dataclasses import dataclass
from typing import Optional

import gradio as gr
import torch

from library import strategy_zimage, zimage_train_utils, zimage_utils
from library.device_utils import clean_memory_on_device, get_preferred_device, init_ipex
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

init_ipex()


def parse_layer_spec(text: Optional[str]):
    if text is None:
        return None
    text = text.strip()
    if text in ("", "all"):
        return None
    parts = [p for p in text.replace(",", " ").split() if p]
    indices = []
    for part in parts:
        if "-" in part:
            start, end = part.split("-", 1)
            start_i = int(start)
            end_i = int(end)
            if end_i < start_i:
                start_i, end_i = end_i, start_i
            indices.extend(range(start_i, end_i + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


@dataclass(frozen=True)
class LoadedConfig:
    model_path: str
    vae_path: str
    text_encoder_path: str
    tokenizer_path: str
    dtype_name: str
    gate_type: str
    gate_layers: Optional[tuple[int, ...]]
    gate_layers_noise_refiner: Optional[tuple[int, ...]]
    gate_layers_context_refiner: Optional[tuple[int, ...]]
    max_token_length: int
    disable_chat_template: bool


class ModelManager:
    def __init__(self, device: torch.device):
        self.device = device
        self.lock = threading.Lock()
        self.loaded_config: Optional[LoadedConfig] = None
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenize_strategy = None
        self.encoding_strategy = None

    def _dtype_from_name(self, dtype_name: str) -> torch.dtype:
        if dtype_name == "fp16":
            return torch.float16
        if dtype_name == "bf16":
            return torch.bfloat16
        return torch.float32

    def unload(self):
        with self.lock:
            self.transformer = None
            self.vae = None
            self.text_encoder = None
            self.tokenize_strategy = None
            self.encoding_strategy = None
            self.loaded_config = None
            if torch.cuda.is_available():
                clean_memory_on_device(self.device)

    def ensure_loaded(
        self,
        model_path: str,
        vae_path: str,
        text_encoder_path: str,
        tokenizer_path: str,
        dtype_name: str,
        gate_type: str,
        gate_layers: Optional[str],
        gate_layers_noise_refiner: Optional[str],
        gate_layers_context_refiner: Optional[str],
        max_token_length: int,
        disable_chat_template: bool,
    ) -> str:
        gate_layers_tuple = tuple(parse_layer_spec(gate_layers) or []) or None
        gate_layers_noise_tuple = tuple(parse_layer_spec(gate_layers_noise_refiner) or []) or None
        gate_layers_context_tuple = tuple(parse_layer_spec(gate_layers_context_refiner) or []) or None

        config = LoadedConfig(
            model_path=model_path,
            vae_path=vae_path,
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path or text_encoder_path,
            dtype_name=dtype_name,
            gate_type=gate_type,
            gate_layers=gate_layers_tuple,
            gate_layers_noise_refiner=gate_layers_noise_tuple,
            gate_layers_context_refiner=gate_layers_context_tuple,
            max_token_length=max_token_length,
            disable_chat_template=disable_chat_template,
        )

        with self.lock:
            if self.loaded_config == config and self.transformer is not None:
                return f"Model already loaded on {self.device}."

            dtype = self._dtype_from_name(dtype_name)
            logger.info("Loading Z-Image models to device %s", self.device)

            self.transformer = zimage_utils.load_transformer(
                model_path,
                dtype,
                self.device,
                gate_type=gate_type,
            )
            if any(v is not None for v in (gate_layers_tuple, gate_layers_noise_tuple, gate_layers_context_tuple)):
                if hasattr(self.transformer, "set_gate_layers"):
                    self.transformer.set_gate_layers(
                        layer_ids=list(gate_layers_tuple) if gate_layers_tuple is not None else None,
                        noise_refiner_ids=list(gate_layers_noise_tuple) if gate_layers_noise_tuple is not None else None,
                        context_refiner_ids=list(gate_layers_context_tuple) if gate_layers_context_tuple is not None else None,
                    )

            self.vae = zimage_utils.load_vae(vae_path, dtype, self.device)
            self.text_encoder = zimage_utils.load_text_encoder(text_encoder_path, dtype, self.device)

            self.transformer.eval()
            self.vae.eval()
            self.text_encoder.eval()

            self.tokenize_strategy = strategy_zimage.ZImageTokenizeStrategy(
                tokenizer_path or text_encoder_path,
                max_length=max_token_length,
                tokenizer_cache_dir=None,
                apply_chat_template=not disable_chat_template,
            )
            self.encoding_strategy = strategy_zimage.ZImageTextEncodingStrategy()
            self.loaded_config = config
            return f"Loaded model to {self.device} with dtype={dtype_name}, gate_type={gate_type}."


def generate_single_image(
    manager: ModelManager,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    steps: int,
    width: int,
    height: int,
    seed: int,
    discrete_flow_shift: float,
    progress: gr.Progress,
):
    transformer = manager.transformer
    vae = manager.vae
    text_encoder = manager.text_encoder
    tokenize_strategy = manager.tokenize_strategy
    encoding_strategy = manager.encoding_strategy

    dtype = next(transformer.parameters()).dtype
    device = manager.device

    height = max(64, int(height) - int(height) % 16)
    width = max(64, int(width) - int(width) % 16)

    prompt_embeds = None
    prompt_mask = None
    negative_embeds = None
    negative_mask = None
    latents = None
    decoded = None

    try:
        with torch.inference_mode():
            prompt_embeds, prompt_mask = zimage_train_utils._encode_prompt(
                tokenize_strategy,
                encoding_strategy,
                text_encoder,
                prompt,
                None,
                device,
                dtype,
            )

            patch_size = transformer.all_patch_size[0] if hasattr(transformer, "all_patch_size") else 2
            image_sequence_length = (height // 8 // patch_size) * (width // 8 // patch_size)
            prompt_embeds, prompt_mask = zimage_train_utils._trim_pad_embeds_and_mask(
                image_sequence_length, prompt_embeds, prompt_mask
            )
            prompt_embeds = prompt_embeds.to(dtype=dtype)

            do_cfg = guidance_scale > 1.0
            if do_cfg:
                negative_embeds, negative_mask = zimage_train_utils._encode_prompt(
                    tokenize_strategy,
                    encoding_strategy,
                    text_encoder,
                    negative_prompt or "",
                    None,
                    device,
                    dtype,
                )
                negative_embeds, negative_mask = zimage_train_utils._trim_pad_embeds_and_mask(
                    image_sequence_length, negative_embeds, negative_mask
                )
                negative_embeds = negative_embeds.to(dtype=dtype)
            else:
                negative_embeds = None
                negative_mask = None

            generator = torch.Generator(device=device).manual_seed(seed)
            latents = torch.randn(
                (1, getattr(transformer, "in_channels", 16), height // 8, width // 8),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )

            timesteps, sigmas = zimage_train_utils._get_timesteps_sigmas(int(steps), float(discrete_flow_shift))
            timesteps = timesteps.to(device)
            sigmas = sigmas.to(device)

            step_iter = progress.tqdm(range(len(timesteps)), desc=f"Seed {seed}", total=len(timesteps), unit="step")
            with torch.autocast(device_type=device.type, dtype=dtype):
                for i in step_iter:
                    t = timesteps[i]
                    timestep = t.expand(latents.shape[0])
                    timestep = (1000 - timestep) / 1000

                    latent_model_input = latents.to(dtype).unsqueeze(2)
                    model_out = transformer(x=latent_model_input, t=timestep, cap_feats=prompt_embeds, cap_mask=prompt_mask)

                    if do_cfg:
                        neg_out = transformer(
                            x=latent_model_input,
                            t=timestep,
                            cap_feats=negative_embeds,
                            cap_mask=negative_mask,
                        )
                        noise_pred = model_out + guidance_scale * (model_out - neg_out)
                    else:
                        noise_pred = model_out

                    noise_pred = -noise_pred.squeeze(2)
                    latents = zimage_train_utils._step(noise_pred.to(torch.float32), latents, sigmas, i)

                latents = latents.to(vae.dtype)
                latents = zimage_train_utils._unscale_latents(latents, vae)
                decoded = zimage_train_utils._decode_latents(vae, latents)
                image = zimage_train_utils._latents_to_pil(decoded)

        return image
    finally:
        del prompt_embeds, prompt_mask, negative_embeds, negative_mask, latents, decoded
        if torch.cuda.is_available():
            clean_memory_on_device(device)


def build_demo(defaults):
    device = get_preferred_device()
    manager = ModelManager(device)

    def load_model():
        return manager.ensure_loaded(
            defaults["model_path"],
            defaults["vae_path"],
            defaults["text_encoder_path"],
            defaults["tokenizer_path"],
            defaults["dtype_name"],
            defaults["gate_type"],
            defaults["gate_layers"],
            defaults["gate_layers_noise_refiner"],
            defaults["gate_layers_context_refiner"],
            int(defaults["max_token_length"]),
            bool(defaults["disable_chat_template"]),
        )

    def unload_model():
        manager.unload()
        return "Model unloaded."

    def generate(
        prompt,
        negative_prompt,
        guidance_scale,
        batch_size,
        steps,
        resolution_preset,
        width,
        height,
        seed,
        discrete_flow_shift,
        progress=gr.Progress(track_tqdm=True),
    ):
        load_message = manager.ensure_loaded(
            defaults["model_path"],
            defaults["vae_path"],
            defaults["text_encoder_path"],
            defaults["tokenizer_path"],
            defaults["dtype_name"],
            defaults["gate_type"],
            defaults["gate_layers"],
            defaults["gate_layers_noise_refiner"],
            defaults["gate_layers_context_refiner"],
            int(defaults["max_token_length"]),
            bool(defaults["disable_chat_template"]),
        )

        if resolution_preset and resolution_preset != "Custom":
            width_str, height_str = resolution_preset.split("x")
            width = int(width_str)
            height = int(height_str)

        batch_size = max(1, int(batch_size))
        steps = max(1, int(steps))
        if seed in (None, "", -1):
            start_seed = random.randint(0, 2**31 - 1)
        else:
            start_seed = int(seed)

        images = []
        seed_list = []
        for batch_index in range(batch_size):
            current_seed = start_seed + batch_index
            progress((batch_index, batch_size), desc=f"Batch {batch_index + 1}/{batch_size}", unit="img")
            image = generate_single_image(
                manager=manager,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=float(guidance_scale),
                steps=steps,
                width=int(width),
                height=int(height),
                seed=current_seed,
                discrete_flow_shift=float(discrete_flow_shift),
                progress=progress,
            )
            images.append(image)
            seed_list.append(current_seed)

        status = (
            f"{load_message}\n"
            f"Generated {len(images)} image(s) on {manager.device}.\n"
            f"Seeds: {', '.join(str(s) for s in seed_list)}\n"
            f"Resolution: {int(width) - int(width) % 16}x{int(height) - int(height) % 16}"
        )
        return images, status

    with gr.Blocks(title="Z-Image Gate Inference") as demo:
        gr.Markdown("## Z-Image Gate Inference")
        gr.Markdown("Models stay loaded on device until config changes or you click unload.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    "\n".join(
                        [
                            f"`model`: {defaults['model_path']}",
                            f"`vae`: {defaults['vae_path']}",
                            f"`text_encoder`: {defaults['text_encoder_path']}",
                            f"`tokenizer`: {defaults['tokenizer_path']}",
                            f"`dtype`: {defaults['dtype_name']}",
                            f"`gate_type`: {defaults['gate_type']}",
                            f"`gate_layers`: {defaults['gate_layers'] or 'default'}",
                        ]
                    )
                )
                with gr.Row():
                    load_btn = gr.Button("Load Model", variant="primary")
                    unload_btn = gr.Button("Unload")

            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Positive Prompt", lines=6, value="A photo of a cat")
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=4, value="")
                with gr.Row():
                    guidance_scale = gr.Slider(label="CFG", minimum=0.0, maximum=12.0, step=0.1, value=4.0)
                    batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=8, step=1, value=1)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, step=1, value=30)
                with gr.Row():
                    resolution_preset = gr.Dropdown(
                        label="Resolution Preset",
                        choices=["Custom", "512x512", "768x768", "1024x1024", "1024x1536", "1536x1024"],
                        value="1024x1024",
                    )
                    width = gr.Slider(label="Width", minimum=256, maximum=2048, step=16, value=1024)
                    height = gr.Slider(label="Height", minimum=256, maximum=2048, step=16, value=1024)
                with gr.Row():
                    seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    discrete_flow_shift = gr.Slider(label="Flow Shift", minimum=0.5, maximum=5.0, step=0.1, value=3.0)
                generate_btn = gr.Button("Generate", variant="primary")
                gallery = gr.Gallery(label="Images", columns=2, preview=True, height="auto")
                status = gr.Textbox(label="Status", lines=6, interactive=False)

        load_btn.click(load_model, outputs=status)
        unload_btn.click(unload_model, outputs=status)
        generate_btn.click(
            generate,
            inputs=[
                prompt,
                negative_prompt,
                guidance_scale,
                batch_size,
                steps,
                resolution_preset,
                width,
                height,
                seed,
                discrete_flow_shift,
            ],
            outputs=[gallery, status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/data/models/zimage_fullgate-000003.safetensors")
    parser.add_argument("--vae", type=str, default="/data/models/Z-Image/vae")
    parser.add_argument("--text_encoder", type=str, default="/data/models/Z-Image/text_encoder")
    parser.add_argument("--tokenizer", type=str, default="/data/models/Z-Image/tokenizer")
    parser.add_argument("--gate_type", type=str, default="elementwise", choices=["headwise", "elementwise", "none"])
    parser.add_argument("--gate_layers", type=str, default="1-22")
    parser.add_argument("--gate_layers_noise_refiner", type=str, default="")
    parser.add_argument("--gate_layers_context_refiner", type=str, default="")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo(
        {
            "model_path": args.pretrained_model_name_or_path,
            "vae_path": args.vae,
            "text_encoder_path": args.text_encoder,
            "tokenizer_path": args.tokenizer,
            "dtype_name": args.dtype,
            "gate_type": args.gate_type,
            "gate_layers": args.gate_layers,
            "gate_layers_noise_refiner": args.gate_layers_noise_refiner,
            "gate_layers_context_refiner": args.gate_layers_context_refiner,
            "max_token_length": 512,
            "disable_chat_template": False,
        }
    )
    demo.queue(default_concurrency_limit=1).launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
