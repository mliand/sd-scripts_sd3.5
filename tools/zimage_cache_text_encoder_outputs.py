# cache text encoder outputs to disk for Z-Image

import argparse

from accelerate.utils import set_seed

from library import config_util, strategy_base, strategy_zimage, train_util, zimage_utils
from library.config_util import ConfigSanitizer, BlueprintGenerator
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


def cache_to_disk(args: argparse.Namespace) -> None:
    setup_logging(args, reset=True)
    train_util.prepare_dataset_args(args, True)
    train_util.enable_high_vram(args)

    args.cache_text_encoder_outputs = True
    args.cache_text_encoder_outputs_to_disk = True

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer_id = args.tokenizer or args.text_encoder
    tokenize_strategy = strategy_zimage.ZImageTokenizeStrategy(
        tokenizer_id,
        max_length=args.max_token_length,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
        apply_chat_template=not args.disable_chat_template,
    )
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

    text_encoder_outputs_caching_strategy = strategy_zimage.ZImageTextEncoderOutputsCachingStrategy(
        True,
        args.text_encoder_batch_size,
        args.skip_cache_check,
        is_partial=False,
        max_length=args.max_token_length,
    )
    strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_outputs_caching_strategy)

    text_encoding_strategy = strategy_zimage.ZImageTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    # dataset setup
    use_user_config = args.dataset_config is not None
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if use_user_config:
            logger.info(f"Loading dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "reg_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            use_dreambooth_method = args.in_json is None
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, _ = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)

    logger.info("prepare accelerator")
    args.deepspeed = False
    accelerator = train_util.prepare_accelerator(args)

    weight_dtype, _ = train_util.prepare_dtype(args)

    if args.text_encoder is None:
        raise ValueError("--text_encoder is required for Z-Image text encoder caching.")

    logger.info("load Z-Image text encoder")
    text_encoder = zimage_utils.load_text_encoder(args.text_encoder, weight_dtype, accelerator.device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    train_dataset_group.new_cache_text_encoder_outputs([text_encoder], accelerator)

    accelerator.wait_for_everyone()
    accelerator.print("Finished caching Z-Image text encoder outputs to disk.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_training_arguments(parser, True)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_masked_loss_arguments(parser)
    config_util.add_config_arguments(parser)
    train_util.add_dit_training_arguments(parser)

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
        "--disable_chat_template",
        action="store_true",
        help="do not apply tokenizer chat template for prompts",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    cache_to_disk(args)
