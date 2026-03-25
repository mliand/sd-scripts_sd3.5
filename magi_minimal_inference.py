import argparse

from library.magi_inference_utils import add_common_inference_arguments, generate_video, prepare_inference_context
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal inference for daVinci-MagiHuman base model in sd-scripts.")
    add_common_inference_arguments(parser)
    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()

    ctx = prepare_inference_context(args)
    generate_video(ctx, args)


if __name__ == "__main__":
    main()

