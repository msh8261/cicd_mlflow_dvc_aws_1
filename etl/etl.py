"""module for data processing with ETL."""
import logging
import os
import string
import sys

import pandas as pd
from prefect import flow, task

from utils.data_utils import get_image_df
from utils.helper import create_parent_directory, load_config

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
# sys.path.insert(0, "..")

logging.basicConfig(level=logging.NOTSET)


@task
def extract_data(path: string) -> pd.DataFrame:
    """Load data from."""
    logging.info("Extracting data...")
    df = get_image_df(path)
    return df


@task
def transform_data() -> None:
    """Transform data."""
    logging.info("Transforming data...")


@task
def load_data(df: pd.DataFrame, path: string, name_file: string) -> None:
    """Load data begin."""
    logging.info("data loaded to directory.")
    df.to_csv(os.path.join(path, name_file))


@flow
def main(config) -> None:
    """ETL data."""
    logging.info("ETL starting...")
    os.makedirs(config.data.path_dst, exist_ok=True)
    train_df = extract_data(config.data.train.path)
    val_df = extract_data(config.data.val.path)
    test_df = extract_data(config.data.test.path)
    transform_data()
    load_data(train_df, config.data.path_dst, config.data.train.name)
    load_data(val_df, config.data.path_dst, config.data.val.name)
    load_data(test_df, config.data.path_dst, config.data.test.name)
    message = "ETL ended."
    logging.info(message)


if __name__ == "__main__":
    config = load_config.fn()
    main(config)
