#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import argparse
import logging
import os
import wandb
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="download_file")
    run.config.update(args)

    logger.info(f"Returning sample {args.sample}")
    logger.info(f"Uploading {args.artifact_name} to Weights & Biases")
    log_artifact(
        artifact_name=args.artifact_name,
        artifact_type=args.artifact_type,
        artifact_description="Raw file as downloaded",
        filename=os.path.join("data", args.sample),
        wandb_run=run
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download URL to a local destination")

    parser.add_argument("sample", type=str, help="Name of the sample to download")

    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")

    parser.add_argument("artifact_type", type=str, help="Output artifact type.")



    args = parser.parse_args()

    go(args)
