"""
Example usage:

python 3_train_policy.py \
--config ../digital_cousins/configs/training/bc_base.json \
--dataset test_demos.hdf5 \
--auto-remove-exp
"""


# Necessary to make sure robomimic registers these modules
import digital_cousins
import omnigibson as og
import json
from robomimic.scripts.train import main as train_main
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
                If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # Output path, to override the one in the config
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="(optional) if provided, override the output folder path defined in the config",
    )

    # force delete the experiment folder if it exists
    parser.add_argument(
        "--auto-remove-exp",
        action='store_true',
        help="force delete the experiment folder if it exists"
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # If args.output is None, make output path be in acdc root dir
    if args.output is None:
        name = config["experiment"]["name"] if args.name is None else args.name
        args.output = f"{digital_cousins.ROOT_DIR}/../training_results"

    # Process dataset
    train_main(args)

    # Shutdown OG at the end
    og.shutdown()

