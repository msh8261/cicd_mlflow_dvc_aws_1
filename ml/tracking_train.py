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


def print_auto_logged_info(r):
    """print information about trained model in mlflow cloud."""
    tags = {
        k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")
    }
    artifacts = [
        f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")
    ]
    logger.info(f"run_id: {r.info.run_id}")
    logger.info(f"artifacts: {artifacts}")
    logger.info(f"params: {r.data.params}")
    logger.info(f"metrics: {r.data.metrics}")
    logger.info(f"tags: {tags}")


def get_dvc_rev(dvc_fp):
    """get the dataset version and return it."""
    with open(dvc_fp) as f:
        s = f.read()
        revs = re.findall(r"rev: (\S+)", s)
    return revs[0] if revs else ""


@task(retries=3, retry_delay_seconds=2)
def get_dataloders(params):
    # initialize the data set splits
    df_train = pd.read_csv(
        os.path.join(config.data.path_dst, config.data.train.name)
    )
    image_size = config.etl.img_size
    transform_train, transform_val = get_preprocessor(
        image_size, config.ml.use_imagenet_pretrained_weights
    )
    dataset_train = XrayDataset(
        df_train,
        transform_train,
        image_size,
        config.ml.use_imagenet_pretrained_weights,
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
    )

    df_val = pd.read_csv(
        os.path.join(config.data.path_dst, config.data.val.name)
    )
    dataset_validation = XrayDataset(
        df_val,
        transform_val,
        image_size,
        config.ml.use_imagenet_pretrained_weights,
    )
    dataloader_validation = DataLoader(
        dataset_validation,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
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
    return dataloader_train, dataloader_validation, dataloader_test


@task
def train(
    params, dataloader_train, dataloader_validation, dataloader_test
) -> None:
    """train a model with best hyperparams and write everything out"""
    # Set random seeds for reproducibility purpose
    seed = 42
    pl.seed_everything(seed=seed, workers=True)

    # Create an experiment. By default, if not specified, the "default" experiment is used. It is recommended to not use
    # the default experiment and explicitly set up your own for better readability and tracking experience.
    client = MlflowClient()
    experiment_name = config.mlflow_experiment_name
    # model_architecture = config.ml.model_architecture
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{config.ml.model_name}_{timestamp}"

    run_name = model_name
    try:
        experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)
    except MlflowException:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    # Fetch experiment metadata information
    logger.info(f"Name: {experiment.name}")
    logger.info(f"Experiment_id: {experiment.experiment_id}")
    logger.info(f"Artifact Location: {experiment.artifact_location}")
    logger.info(f"Tags: {experiment.tags}")
    logger.info(f"Lifecycle_stage: {experiment.lifecycle_stage}")

    os.makedirs(config.model_dir_output, exist_ok=True)
    training_output_dir = os.path.join(config.model_dir_output, model_name)
    checkpoints_dir = os.path.join(training_output_dir, "checkpoints")
    # dataset_dvc_fp = config.dataset_dvc
    # dataset_version = get_dvc_rev(dataset_dvc_fp)

    # model
    model = XrayClassifier(
        imagenet_weights=config.ml.use_imagenet_pretrained_weights,
        n_layers=params["n_layers"],
        dropout=params["dropout"],
        lr=params["lr"],
    )
    monitor = config.ml.monitor
    mode = config.ml.mode
    checkpoint_name_format = "{epoch:03d}_{" + monitor + ":.3f}"

    callbacks = [
        pl.callbacks.model_checkpoint.ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename=checkpoint_name_format,
            monitor=monitor,
            save_last=True,
            save_top_k=-1,  # save all checkpoints
            mode=mode,
            every_n_epochs=1,
        ),
        pl.callbacks.early_stopping.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=params["early_stopping_patience"],
            verbose=True,
        ),
    ]

    tensorboard_logger = pl_loggers.TensorBoardLogger(
        training_output_dir, name="tensorboard"
    )

    trainer = pl.Trainer(
        gpus=params["gpus"],
        precision=params["precision"],
        max_epochs=params["max_epochs"],
        callbacks=callbacks,
        deterministic=True,
        logger=tensorboard_logger,
    )
    # Activate auto logging for pytorch lightning module
    mlflow.pytorch.autolog()

    # Launch training phase
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=run_name
    ) as run:
        logger.info("tracking uri:", mlflow.get_tracking_uri())
        logger.info("artifact uri:", mlflow.get_artifact_uri())
        logger.info("start training")

        # # save dataset's dvc file
        # mlflow.log_artifact(dataset_dvc_fp)

        trainer.fit(model, dataloader_train, dataloader_validation)
        mlflow.log_artifacts(training_output_dir)

        # log training parameters
        mlflow.log_params(params)
        _, y_test = dataloader_test[:]
        y_pred = trainer.predict(model, dataloaders=dataloader_test)
        # Log desired metrics
        mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("rmse", sqrt(mean_squared_error(y_test, y_pred)))
        mlflow.log_metric("r2", r2_score(y_test, y_pred))

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


@flow
def main():
    """The main training pipeline"""

    params = config.ml.best_params

    dataloader_train, dataloader_validation, dataloader_test = get_dataloders(
        params
    )

    train(params, dataloader_train, dataloader_validation, dataloader_test)


if __name__ == "__main__":
    main()
