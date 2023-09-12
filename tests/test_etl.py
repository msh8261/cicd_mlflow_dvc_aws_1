import os
import sys
sys.path.insert(0, '..')
from etl.etl import  extract_data, transform_data, load_data ,data_etl



def test_extract_data():
    '''test extracting data'''
    extract_data()
    assert 1 == 1

def test_transform_data():
    '''test transforming data'''
    transform_data()
    assert 1 == 1


def test_load_data():
    '''test loading data'''
    load_data('data')
    assert 1 == 1


def test_data_etl():
    '''test etl pipline'''
    data_etl()
    assert 1 == 1
