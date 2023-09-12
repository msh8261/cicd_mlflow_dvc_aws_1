import os
import sys
sys.path.insert(0, '..')
from datetime import datetime
import pandas as pd
from prefect import task, flow
import logging
from typing import Tuple, List
import hydra
from hydra import initialize, compose


logging.basicConfig(level=logging.NOTSET)

# global initialization
with initialize(version_base=None, config_path='config'):
    config = compose(config_name='main', overrides=['etl=etl1'])


@task
def extract_data():
    '''Load data from'''
    logging.info('Extracting data')
    return None


@task
def transform_data():
    '''Transform data'''
    logging.info('Transforming data')
    return None


@task
def load_data(data):
    '''Loading data to ...'''
    logging.info('data loaded to directory.')


@flow
def data_etl():
    '''ETL data'''
    logging.info('ETL starting...')
    raw_data = extract_data()
    data = transform_data()
    load_data(data)
    logging.info('ETL ended.')



if __name__ == "__main__":
    data_etl()
