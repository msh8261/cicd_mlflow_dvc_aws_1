"""module for test etl module."""
import os
import sys

import pandas as pd

from etl.etl import extract_data, load_data, transform_data
from utils.helper import create_parent_directory, load_config

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

# sys.path.insert(0, "..")

config = load_config.fn()


def test_path_exist():
    """test path of files exist."""
    assert os.path.exists(config.data.path)
    assert os.path.exists(config.data.train.path)
    assert os.path.exists(config.data.val.path)
    assert os.path.exists(config.data.test.path)


def test_file_name():
    """test the name and format of data files."""
    assert config.data.train.name == "train.csv"
    assert config.data.val.name == "val.csv"
    assert config.data.test.name == "test.csv"


def test_extract_data():
    """test extract data."""
    df1 = extract_data.fn(config.data.train.path)
    df2 = extract_data.fn(config.data.val.path)
    df3 = extract_data.fn(config.data.test.path)
    assert type(df1) == pd.DataFrame
    assert type(df2) == pd.DataFrame
    assert type(df3) == pd.DataFrame


def test_transform_data():
    """test transform data."""
    data = "ok"
    transform_data.fn()
    assert data == "ok"


def test_load_data():
    """test load data."""
    data = [["C:/Users/data/dummy/images/img.jpeg", 1]]
    dummy_df = pd.DataFrame(data, columns=["img_path", "label"])
    file_dir = "./tests/"
    file_name = "dummy_data.csv"
    load_data.fn(dummy_df, file_dir, file_name)
    assert os.path.exists(os.path.join(file_dir, file_name))
    df = pd.read_csv(os.path.join(file_dir, file_name))
    os.remove(os.path.join(file_dir, file_name))
    assert type(df) == pd.DataFrame
    assert df["img_path"][0] == "C:/Users/data/dummy/images/img.jpeg"
    assert df["label"][0] == 1
