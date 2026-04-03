import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import library.train_util as train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = [
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
    ".avi",
    ".MP4",
    ".WEBM",
    ".MOV",
    ".MKV",
    ".AVI",
]


def _expand_path(path: str | None) -> str | None:
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _glob_media_pathlib(dir_path: Path, recursive: bool) -> List[Path]:
    media_paths = set(train_util.glob_images_pathlib(dir_path, recursive))
    glob_fn = dir_path.rglob if recursive else dir_path.glob
    for ext in VIDEO_EXTENSIONS:
        media_paths.update(glob_fn("*" + ext))
    return sorted(media_paths)


def main(args):
    assert not args.recursive or (
        args.recursive and args.full_path
    ), "recursive requires full_path / recursive requires full_path"

    args.train_data_dir = _expand_path(args.train_data_dir)
    args.out_json = _expand_path(args.out_json)
    args.in_json = _expand_path(args.in_json)

    train_data_dir_path = Path(args.train_data_dir)
    media_paths: List[Path] = _glob_media_pathlib(train_data_dir_path, args.recursive)
    logger.info(f"found {len(media_paths)} media files.")

    if args.in_json is None and Path(args.out_json).is_file():
        args.in_json = args.out_json

    if args.in_json is not None:
        logger.info(f"loading existing metadata: {args.in_json}")
        metadata = json.loads(Path(args.in_json).read_text(encoding="utf-8"))
        logger.warning("captions for existing media entries will be overwritten")
    else:
        logger.info("new metadata will be created")
        metadata = {}

    logger.info("merge caption texts to metadata json.")
    skipped = 0
    for media_path in tqdm(media_paths):
        caption_path = media_path.with_suffix(args.caption_extension)
        if not caption_path.exists():
            skipped += 1
            if args.debug:
                logger.warning(f"caption not found, skip: {caption_path}")
            continue

        caption = caption_path.read_text(encoding="utf-8").strip()

        image_key = str(media_path) if args.full_path else media_path.stem
        if image_key not in metadata:
            metadata[image_key] = {}

        metadata[image_key]["caption"] = caption
        if args.debug:
            logger.info(f"{image_key} {caption}")

    if skipped:
        logger.warning(f"skipped {skipped} media files without captions")

    logger.info(f"writing metadata: {args.out_json}")
    Path(args.out_json).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images/videos")
    parser.add_argument("out_json", type=str, help="metadata file to output")
    parser.add_argument(
        "--in_json",
        type=str,
        help="metadata file to input (if omitted and out_json exists, existing out_json is read)",
    )
    parser.add_argument(
        "--caption_extention",
        type=str,
        default=None,
        help="legacy alias of --caption_extension for backward compatibility",
    )
    parser.add_argument(
        "--caption_extension",
        type=str,
        default=".caption",
        help="extension of caption file",
    )
    parser.add_argument(
        "--full_path",
        action="store_true",
        help="use full path as metadata key",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="recursively scan child folders under train_data_dir",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention

    main(args)
