import argparse

from library.magi_inference_utils import (
    add_common_inference_arguments,
    add_lora_inference_arguments,
    apply_lora_to_wrapper,
    generate_video,
    prepare_inference_context,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal inference for daVinci-MagiHuman + LoRA in sd-scripts.")
    add_common_inference_arguments(parser)
    add_lora_inference_arguments(parser)
    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()

    ctx = prepare_inference_context(args)
    apply_lora_to_wrapper(ctx.wrapper, args.lora_weights, args.merge_lora_weights)
    generate_video(ctx, args)


if __name__ == "__main__":
    main()

