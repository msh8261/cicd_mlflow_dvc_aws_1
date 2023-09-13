"""module for data processing with ETL."""
import logging
import sys
from typing import List, Tuple

# import hydra
from hydra import compose, initialize
# import pandas as pd
from prefect import flow, task

# import os
sys.path.insert(0, "..")

logging.basicConfig(level=logging.NOTSET)

# global initialization
with initialize(version_base=None, config_path="../config"):
    config = compose(config_name="main", overrides=["etl=etl1"])
    # print(OmegaConf.to_yaml(config))


@task
def extract_data() -> List:
    """Load data from."""
    logging.info("Extracting data...")
    return ["yes"]


@task
def transform_data() -> Tuple:
    """Transform data."""
    logging.info("Transforming data...")
    return ("yes", "no")


@task
def load_data(data):
    """Load data begin."""
    logging.info("data loaded to directory.")
    return data


@flow
def data_etl():
    """ETL data."""
    logging.info("ETL starting...")
    data = extract_data()
    data = transform_data()
    data = load_data(data)
    logging.info("ETL ended.")


if __name__ == "__main__":
    data_etl()
