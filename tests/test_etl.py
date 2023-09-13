"""module for test etl module."""
import sys

# import os
from etl.etl import data_etl, extract_data, load_data, transform_data

sys.path.insert(0, "..")


def test_extract_data():
    """test extract data."""
    data = extract_data.fn()
    assert data == ["yes"]


def test_transform_data():
    """test transform data."""
    data = transform_data.fn()
    assert data == ("yes", "no")


def test_load_data():
    """test load data."""
    data = load_data.fn("data")
    assert data == "data"


def test_data_etl():
    """test etl pipline."""
    data_etl()
    # assert 'yes' == 'yes'
