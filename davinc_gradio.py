import argparse
import os
import random
import tempfile
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import gradio as gr
import torch

from library.magi_inference_utils import generate_video, prepare_inference_context
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


MAX_SIDE = 640
MAX_FRAMES = 245
MIN_SIDE = 64
SIDE_STEP = 16


def _clamp_resolution(width: int, height: int) -> tuple[int, int]:
    width = max(MIN_SIDE, int(width))
    height = max(MIN_SIDE, int(height))

    longest = max(width, height)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / float(longest)
        width = max(MIN_SIDE, int(width * scale))
        height = max(MIN_SIDE, int(height * scale))

    width -= width % SIDE_STEP
    height -= height % SIDE_STEP
    width = max(MIN_SIDE, width)
    height = max(MIN_SIDE, height)
    return width, height


def _normalize_num_frames(num_frames: int) -> int:
    return max(1, min(int(num_frames), MAX_FRAMES))


@dataclass(frozen=True)
class LoadedConfig:
    pretrained_model_name_or_path: str
    config_load_path: Optional[str]
    vae_model_path: str
    audio_model_path: str
    txt_model_path: str
    device: str
    model_dtype: str
    decode_dtype: str
    cpu_offload: bool
    offload_text_encoder: bool
    offload_vae: bool
    offload_audio_vae: bool


class DavincManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.ctx = None
        self.loaded_config: Optional[LoadedConfig] = None

    def unload(self) -> str:
        with self.lock:
            self.ctx = None
            self.loaded_config = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "模型已卸载。"

    def ensure_loaded(self, config: LoadedConfig) -> str:
        with self.lock:
            if self.ctx is not None and self.loaded_config == config:
                return "模型已加载，复用当前上下文。"

            os.environ["CPU_OFFLOAD"] = "1" if config.cpu_offload else "0"

            args = SimpleNamespace(
                pretrained_model_name_or_path=config.pretrained_model_name_or_path,
                config_load_path=config.config_load_path,
                vae_model_path=config.vae_model_path,
                audio_model_path=config.audio_model_path,
                txt_model_path=config.txt_model_path,
                device=config.device,
                model_dtype=config.model_dtype,
                decode_dtype=config.decode_dtype,
                offload_text_encoder=config.offload_text_encoder,
                offload_vae=config.offload_vae,
                offload_audio_vae=config.offload_audio_vae,
            )

            logger.info("Loading daVinci inference context on device=%s", config.device)
            self.ctx = prepare_inference_context(args)
            self.loaded_config = config
            return "模型加载完成。"


def build_demo(defaults: dict):
    manager = DavincManager()
    output_dir = Path(defaults["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="sky",
        neutral_hue="slate",
    )

    css = """
    .gradio-container {
      background:
        radial-gradient(circle at top right, rgba(96,165,250,0.18), transparent 28%),
        linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
    }
    .davinc-card {
      border: 1px solid rgba(96,165,250,0.18);
      border-radius: 18px;
      background: rgba(255,255,255,0.92);
      box-shadow: 0 12px 32px rgba(37, 99, 235, 0.08);
    }
    """

    def _make_config() -> LoadedConfig:
        return LoadedConfig(
            pretrained_model_name_or_path=defaults["pretrained_model_name_or_path"],
            config_load_path=defaults["config_load_path"],
            vae_model_path=defaults["vae_model_path"],
            audio_model_path=defaults["audio_model_path"],
            txt_model_path=defaults["txt_model_path"],
            device=defaults["device"],
            model_dtype=defaults["model_dtype"],
            decode_dtype=defaults["decode_dtype"],
            cpu_offload=bool(defaults["cpu_offload"]),
            offload_text_encoder=True,
            offload_vae=True,
            offload_audio_vae=True,
        )

    def load_model() -> str:
        config = _make_config()
        return manager.ensure_loaded(config)

    def unload_model() -> str:
        return manager.unload()

    def generate(
        prompt: str,
        negative_prompt: str,
        first_frame,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        progress=gr.Progress(track_tqdm=True),
    ):
        if prompt is None or prompt.strip() == "":
            raise gr.Error("Prompt 不能为空。")

        width, height = _clamp_resolution(int(width), int(height))
        num_frames = _normalize_num_frames(num_frames)
        seed = random.randint(0, 2**31 - 1) if seed in (None, "", -1) else int(seed)

        load_message = load_model()
        config = manager.loaded_config
        if manager.ctx is None or config is None:
            raise gr.Error("模型尚未成功加载。")

        image_path = None
        temp_image_path = None
        try:
            if first_frame is not None:
                fd, temp_name = tempfile.mkstemp(prefix="davinc_first_frame_", suffix=".png", dir=str(output_dir))
                os.close(fd)
                temp_image_path = temp_name
                first_frame.save(temp_image_path)
                image_path = temp_image_path

            args = SimpleNamespace(
                prompt=prompt.strip(),
                negative_prompt=(negative_prompt or "").strip(),
                image_path=image_path,
                guidance_scale=5.0,
                audio_guidance_scale=None,
                low_t_video_guidance=2.0,
                num_inference_steps=40,
                discrete_flow_shift=5.0,
                seed=seed,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=24,
                device=config.device,
                model_dtype=config.model_dtype,
                decode_dtype=config.decode_dtype,
                output_dir=str(output_dir),
                output_name="davinc_sample",
                video_only=False,
                txt_model_path=config.txt_model_path,
                offload_text_encoder=True,
                offload_vae=True,
                offload_audio_vae=True,
            )

            progress(0, desc="开始推理")
            with manager.lock:
                video_path = generate_video(manager.ctx, args)

            status = (
                f"{load_message}\n"
                f"推理完成\n"
                f"Seed: {seed}\n"
                f"Resolution: {width}x{height}\n"
                f"Frames: {num_frames}\n"
                f"CPU offload: {'on' if defaults['cpu_offload'] else 'off'}"
            )
            return video_path, status
        except gr.Error:
            raise
        except Exception as e:
            logger.exception("davinc gradio generation failed")
            raise gr.Error(f"推理失败: {e}\n\n{traceback.format_exc(limit=2)}")
        finally:
            if temp_image_path is not None:
                try:
                    os.remove(temp_image_path)
                except OSError:
                    pass

    with gr.Blocks(title="daVinci Inference", theme=theme, css=css) as demo:
        gr.Markdown(
            """
            # daVinci 推理
            蓝白主题的最小推理界面。模型只加载一次，请求串行排队，避免多任务把显存打满。
            """
        )

        with gr.Row(elem_classes=["davinc-card"]):
            with gr.Column(scale=1):
                gr.Markdown(
                    "\n".join(
                        [
                            f"`ckpt`: {defaults['pretrained_model_name_or_path']}",
                            f"`vae`: {defaults['vae_model_path']}",
                            f"`audio`: {defaults['audio_model_path']}",
                            f"`text`: {defaults['txt_model_path']}",
                            f"`dtype`: {defaults['model_dtype']}/{defaults['decode_dtype']}",
                            f"`device`: {defaults['device']}",
                            f"`output_dir`: {defaults['output_dir']}",
                        ]
                    )
                )
                with gr.Row():
                    load_btn = gr.Button("加载模型", variant="primary")
                    unload_btn = gr.Button("卸载模型")
                status = gr.Textbox(label="状态", lines=8, interactive=False)

            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", lines=8, placeholder="描述动作、人物、镜头语言")
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=4, placeholder="不希望出现的内容")
                first_frame = gr.Image(label="First Frame", type="pil")
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=MIN_SIDE, maximum=MAX_SIDE, step=SIDE_STEP, value=480)
                    height = gr.Slider(label="Height", minimum=MIN_SIDE, maximum=MAX_SIDE, step=SIDE_STEP, value=272)
                with gr.Row():
                    num_frames = gr.Slider(label="Frames", minimum=1, maximum=MAX_FRAMES, step=1, value=81)
                    seed = gr.Number(label="Seed (-1 随机)", value=-1, precision=0)
                generate_btn = gr.Button("生成视频", variant="primary")
                video = gr.Video(label="Output Video", interactive=False)

        load_btn.click(
            load_model,
            outputs=status,
        )
        unload_btn.click(unload_model, outputs=status)
        generate_btn.click(
            generate,
            inputs=[
                prompt,
                negative_prompt,
                first_frame,
                width,
                height,
                num_frames,
                seed,
            ],
            outputs=[video, status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio app for daVinci inference.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--config_load_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, required=True)
    parser.add_argument("--audio_model_path", type=str, required=True)
    parser.add_argument("--txt_model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--decode_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU_OFFLOAD env for text encoder wrapper.")
    parser.add_argument("--output_dir", type=str, default="gradio_outputs/davinc")
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo(
        {
            "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
            "config_load_path": args.config_load_path,
            "vae_model_path": args.vae_model_path,
            "audio_model_path": args.audio_model_path,
            "txt_model_path": args.txt_model_path,
            "device": args.device,
            "model_dtype": args.model_dtype,
            "decode_dtype": args.decode_dtype,
            "cpu_offload": args.cpu_offload,
            "output_dir": args.output_dir,
        }
    )
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
