# cache latents to disk for Z-Image

import argparse

from accelerate.utils import set_seed
import torch

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

    args.cache_latents = True
    args.cache_latents_to_disk = True

    if args.seed is not None:
        set_seed(args.seed)

    latents_caching_strategy = strategy_zimage.ZImageLatentsCachingStrategy(True, args.vae_batch_size, args.skip_cache_check)
    strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

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
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    if args.vae is None:
        raise ValueError("--vae is required for Z-Image latent caching.")

    logger.info("load Z-Image VAE")
    vae = zimage_utils.load_vae(args.vae, vae_dtype, "cpu")
    vae.to(accelerator.device, dtype=vae_dtype)
    vae.requires_grad_(False)
    vae.eval()

    train_dataset_group.new_cache_latents(vae, accelerator)

    accelerator.wait_for_everyone()
    accelerator.print("Finished caching Z-Image latents to disk.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_training_arguments(parser, True)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_masked_loss_arguments(parser)
    config_util.add_config_arguments(parser)

    parser.add_argument(
        "--vae",
        type=str,
        required=True,
        help="path to Z-Image VAE (diffusers directory or safetensors file)",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    cache_to_disk(args)
