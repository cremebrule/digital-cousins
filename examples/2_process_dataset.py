"""
Example usage:

python 2_process_dataset.py \
--dataset test_demos.hdf5 \
--ratio 0.34 \
--seed 0
"""

from robomimic.scripts.split_train_val import split_train_val_from_hdf5
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="if provided, split the subset of trajectories in the file that correspond to\
            this filter key into a training and validation set of trajectories, instead of\
            splitting the full set of trajectories",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="validation ratio, in (0, 1)"
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed to use for randomization")
    args = parser.parse_args()

    # Set seed deterministically
    np.random.seed(args.seed)

    # Process dataset
    split_train_val_from_hdf5(args.dataset, val_ratio=args.ratio, filter_key=args.filter_key)
