"""soup: A Flower / sklearn app."""

import numpy as np
from datasets import load_dataset
from pandas import read_csv
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging

fds = None  # Cache FederatedDataset
data_path = '/home/elena/ATTRITION/soup/data/dataset-2.csv'


def load_data():
    dataset = read_csv(data_path, delimiter=',')

    dataset = dataset.transpose()
    trg = dataset[-1:]
    trn = dataset[:-1]

    x_train, x_test, y_train, y_test = train_test_split(trn.transpose(), trg.transpose(), test_size=0.2)
    print(x_test)

    return x_train, x_test, y_train, y_test


def get_model_params(model):
    logging.debug("Getting model parameters")
    try:
        params = [model.estimators_]
        print("\nGet model params:", params)
    except Exception as e:
        logging.error("Exception getting model parameters %s", e)
    return params


def set_model_params(model, params):
    logging.debug("Setting model param")
    model.estimators_ = params[0]
    return model


