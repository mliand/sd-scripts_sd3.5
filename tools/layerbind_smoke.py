import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run a quick LayerBind smoke test on top of zimage_minimal_inference.py")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use")
    parser.add_argument("--model", type=str, required=True, help="Path or HF id for Z-Image base transformer")
    parser.add_argument("--vae", type=str, required=True, help="Path to the Z-Image VAE")
    parser.add_argument("--text_encoder", type=str, required=True, help="Path or HF id for the Qwen text encoder")
    parser.add_argument(
        "--layout",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "layerbind_layout_example.json"),
        help="Path to a LayerBind layout JSON file",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Override the scene prompt")
    parser.add_argument("--output_dir", type=str, default="outputs/layerbind_smoke")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    cmd = [
        args.python,
        "zimage_minimal_inference.py",
        "--pretrained_model_name_or_path",
        args.model,
        "--vae",
        args.vae,
        "--text_encoder",
        args.text_encoder,
        "--layerbind_layout",
        args.layout,
        "--layerbind_save_intermediates",
        "--steps",
        str(args.steps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--seed",
        str(args.seed),
        "--output_dir",
        args.output_dir,
    ]

    if args.prompt is not None:
        cmd.extend(["--prompt", args.prompt])
    if args.bf16:
        cmd.append("--bf16")
    if args.fp16:
        cmd.append("--fp16")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
