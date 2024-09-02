import os

import mlflow
from hydra import compose, initialize
from omegaconf import DictConfig


def train_model(config: DictConfig):
    """
    Train a model using the datacomp training script on filtered data.

    Args:
        config (DictConfig): Hydra configuration object containing training parameters.
    """
    # Initialize MLflow
    mlflow.set_tracking_uri(os.environ["MLFLOW"])
    mlflow.start_run()

    # Log parameters
    mlflow.log_params(config)

    # TODO: Implement actual training logic using datacomp training script
    # This is a placeholder for the actual implementation
    print("Training model with datacomp script on filtered data")

    # Log metrics
    # TODO: Log actual metrics from training
    mlflow.log_metric("accuracy", 0.85)

    mlflow.end_run()


if __name__ == "__main__":
    with initialize(config_path="../conf"):
        config = compose(config_name="train_config")
        train_model(config)
