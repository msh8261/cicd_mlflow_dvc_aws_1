"""the whole pipline for train, validataion and test with mlflow."""
import os
import re
import sys
import time
from math import sqrt

import mlflow
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from prefect import flow, task
from pytorch_lightning import loggers as pl_loggers
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from ml.classifier import XrayClassifier
from ml.dataset import XrayDataset
from ml.processor import get_preprocessor
from utils.helper import create_parent_directory, load_config

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

config = load_config.fn()

gpus = 1 if torch.cuda.is_available() else 0


@task(retries=3, retry_delay_seconds=2)
def get_test_dataloder(params):
    """read csv file and return the dataloader"""
    # initialize the data set splits
    image_size = config.etl.img_size
    _, transform_val = get_preprocessor(
        image_size, config.ml.use_imagenet_pretrained_weights
    )
    df_test = pd.read_csv(
        os.path.join(config.data.path_dst, config.data.test.name)
    )
    dataset_test = XrayDataset(
        df_test,
        transform_val,
        image_size,
        config.ml.use_imagenet_pretrained_weights,
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )
    return dataloader_test


@task(log_prints=True)
def update_production_model(client, run, model_name) -> None:
    """get the current trained model and compare it with latest
    production model and updatae the best.
    """
    new_run_id = run.info.run_id
    new_run = client.get_run(new_run_id)
    new_metrics = new_run.data.metrics

    prod_run_id = client.get_latest_versions(
        model_name, stages=["Production"]
    )[0].run_id
    prod_run = client.get_run(prod_run_id)
    prod_metrics = prod_run.data.metrics

    # Collate metrics into DataFrame for comparison
    columns = ["mse", "rmse", "r2"]
    columns = ["version"] + [x for x in sorted(columns)]
    new_vals = ["new"] + [
        new_metrics[m] for m in sorted(new_metrics) if m in columns
    ]
    prod_vals = ["prod"] + [
        prod_metrics[m] for m in sorted(prod_metrics) if m in columns
    ]
    data = [new_vals, prod_vals]

    metrics_df = pd.DataFrame(data, columns=columns)
    new_mse = metrics_df[metrics_df["version"] == "new"]["mse"].values[0]
    new_rmse = metrics_df[metrics_df["version"] == "new"]["rmse"].values[0]
    new_r2 = metrics_df[metrics_df["version"] == "new"]["r2"].values[0]

    prod_mse = metrics_df[metrics_df["version"] == "prod"]["mse"].values[0]
    prod_rmse = metrics_df[metrics_df["version"] == "prod"]["rmse"].values[0]
    prod_r2 = metrics_df[metrics_df["version"] == "prod"]["r2"].values[0]

    # Check new model meets our validation criteria before promoting to production
    if (new_mse < prod_mse) and (new_rmse < prod_rmse) and (new_r2 > prod_r2):
        model_uri = (
            "s3:/bucket-cicd-2023/mlflow-tracking/<>/<>/artifacts/model"
        )
        print("run_id is: ", new_run_id)

        desc = (
            "This model uses mobilenet classifier to classify chest diseases."
        )

        client.create_model_version(
            model_name, model_uri, new_run_id, description=desc
        )
        to_prod_version = client.search_model_versions(
            "run_id='{}'".format(new_run_id)
        )[0].version
        to_archive_version = client.search_model_versions(
            "run_id='{}'".format(prod_run_id)
        )[0].version

        # Transition new model to Production stage
        client.transition_model_version_stage(
            model_name, to_prod_version, "Production"
        )

        # Wait for the transition to complete
        new_prod_version = client.get_model_version(
            model_name, to_prod_version
        )
        while new_prod_version.current_stage != "Production":
            new_prod_version = client.get_model_version(
                model_name, to_prod_version
            )
            print(
                "Transitioning new model... Current model version is: ",
                new_prod_version.current_stage,
            )
            time.sleep(1)

        # Transition old model to Archived stage
        client.transition_model_version_stage(
            model_name, to_archive_version, "Archived"
        )

    else:
        print("no improvement")


@task
def register(params, dataloader_test) -> pd.DataFrame:
    """train a model with best hyperparams and write everything out"""
    # Set random seeds for reproducibility purpose
    seed = 42
    pl.seed_everything(seed=seed, workers=True)

    model_name = config.ml.model_name
    new_stage = "Staging"

    # Create an experiment. By default, if not specified, the "default" experiment is used. It is recommended to not use
    # the default experiment and explicitly set up your own for better readability and tracking experience.
    client = MlflowClient()
    experiment_name = config.mlflow_experiment_name

    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=experiment_id,
        filter_string="metrics.rmse < 7",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.rmse ASC"],
    )
    best_run = runs[
        runs.index(max([run.data.metrics["rmse"] for run in runs]))
    ]

    print(
        f"run id: {best_run.info.run_id}, rmse: {best_run.data.metrics['rmse']:.4f}"
    )

    model_uri = f"runs:/{best_run.info.run_id}/model"

    mlflow.register_model(model_uri=model_uri, name=model_name)

    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{new_stage}")
    # log training parameters
    mlflow.log_params(params)
    X_test, y_test = dataloader_test[:]
    y_pred = model.predict(X_test)
    # Log desired metrics
    mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
    mlflow.log_metric("rmse", sqrt(mean_squared_error(y_test, y_pred)))
    mlflow.log_metric("r2", r2_score(y_test, y_pred))

    return client, best_run, model_name


@flow
def main():
    """The main training pipeline"""

    params = config.ml.best_params

    dataloader_test = get_test_dataloder(params)

    client, best_run, model_name = register(params, dataloader_test)

    update_production_model(client, best_run, model_name)


if __name__ == "__main__":
    main()
