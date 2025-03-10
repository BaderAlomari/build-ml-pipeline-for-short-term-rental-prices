#!/usr/bin/env python
"""
An example of a step using MLflow and Weights & Biases]: Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd



logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info("downloading artifact.")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    
    logger.info("loading artifact.")
    df = pd.read_csv(artifact_local_path)
    
    logger.info("dropping outliers")
    minimum_price = args.min_price
    maximum_price = args.max_price
    df['last_review'] = pd.to_datetime(df['last_review'])
    idx = df['price'].between(minimum_price, maximum_price)
    df = df[idx].copy()
    
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
 
    output_file_name = "clean_sample"
    logger.info(f"saving the df to {output_file_name}")
    df.to_csv(f"{output_file_name}.csv", index=False)
    
    logger.info("init artifact")
    artifact = wandb.Artifact(
    name="clean_sample.csv",
    type=args.output_type
    # description=args.output_description,
    )
    artifact.add_file(f"{output_file_name}.csv")
    run.log_artifact(artifact)
    
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="input file name",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="output file name",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="type of output data",
        required=True
    )

    # parser.add_argument(
    #     "--output_description", 
    #     type=str,
    #     help="cleaned the data successfully",
    #     required=False
    # )

    parser.add_argument(
        "--min_price", 
        type=int,
        help="the minimum price to compare",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=int,
        help="the maximum price to compare",
        required=True
    )


    args = parser.parse_args()

    go(args)
