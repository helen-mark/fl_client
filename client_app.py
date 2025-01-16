"""soup: A Flower / sklearn app."""

import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import log_loss

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from task import (
    get_model_params,
    load_data,
    set_model_params,
)
import flwr as fl
import numpy as np
import logging

# logging.basicConfig(level=logging.DEBUG)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        try:
            print("\nInit client model")
            self.model = RandomForestRegressor(n_estimators=10, max_features='sqrt')
            print("\nmodel.get_params() after init:", self.model.get_params())
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

        except Exception as e:
            logging.error("Error initializing FlowerClient: %s", e)

    def fit(self, parameters, config):
        logging.info("Fitting...")
        set_model_params(self.model, parameters)
        print("model params before fitting:", self.model.get_params())
        print("\nfitting...")
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        print("\nModel attrs after fitting:", self.model.__dict__)

        return get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        print("\nModel params before evaluation:", self.model.__dict__)
        set_model_params(self.model, parameters)
        print(parameters)
        print("Client: Starting evaluation.")
        predictions = self.model.predict(self.x_test)
        loss = np.mean((predictions - self.y_test) ** 2)
        print(f"Client: Evaluation completed with loss: {loss}.")

        return loss, len(self.X_test), {"loss": loss}


#def client_fn(context: Context):

X_train, X_test, y_train, y_test = load_data()
model = RandomForestRegressor(n_estimators=10, max_features='sqrt')

logging.info("Starting Flower client.")
fl.client.start_numpy_client(server_address="192.168.2.31:50057", client=FlowerClient(model, X_train, X_test, y_train, y_test))
