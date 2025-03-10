#!/usr/bin/env python
"""
This script splits the provided dataframe in test and remainder
"""
import argparse
import logging
import pandas as pd
import wandb
import tempfile
from sklearn.model_selection import train_test_split
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    logger.info(f"Fetching artifact {args.input}")
    artifact_local_path = run.use_artifact(args.input).file()

    df = pd.read_csv(artifact_local_path)

    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    # Save to output files
    for df_split, split_name in zip([trainval, test], ['trainval', 'test']):
        logger.info(f"Uploading {split_name}_data.csv dataset")
        filename = f"{split_name}_data.csv"
        df_split.to_csv(filename, index=False)
    
        log_artifact(
        filename,
        f"{split_name}_data",
        f"{split_name} split of dataset",
        filename,
        run,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size", type=float, help="Size of the test split. Fraction of the dataset, or number of items"
    )

    parser.add_argument(
        "--random_seed", type=int, help="Seed for random number generator", default=42, required=False
    )

    parser.add_argument(
        "--stratify_by", type=str, help="Column to use for stratification", default='none', required=False
    )

    args = parser.parse_args()

    go(args)
