import os
from typing import List

import mlflow
import optuna
import ray
from eval_lib import evaluate_model
from filter_lib import (
    CLIPScoreFilter,
    DPCLIPScoreFilter,
    DPNegCLIPLossFilter,
    DPTRAKFilter,
    GeneralFilter,
    NegCLIPLossFilter,
    TRAKFilter,
    apply_filters,
)
from hydra import compose, initialize
from omegaconf import DictConfig
from train_lib import train_model


def run_experiment(config: DictConfig):
    """
    Run an experiment with the given configuration.

    Args:
        config (DictConfig): Hydra configuration object containing experiment parameters.
    """
    mlflow.set_tracking_uri(os.environ["MLFLOW"])
    mlflow.set_experiment(config.experiment_name)

    # Initialize Ray
    ray.init(namespace="datacomp", ignore_reinit_error=True)

    with mlflow.start_run():
        mlflow.log_params(config)

        # Apply filters
        data = load_data(
            config.data_path
        )  # TODO: Implement load_data function
        filters = create_filters(config)
        filtered_data = apply_filters(data, filters)

        # Train model
        model = train_model(config, filtered_data)

        # Evaluate model
        text_encoder = load_text_encoder(
            config
        )  # TODO: Implement load_text_encoder function
        evaluate_model(model, text_encoder)


def create_filters(config: DictConfig) -> List[ray.ObjectRef]:
    """
    Create a list of Ray actor handles for filters based on the configuration.

    Args:
        config (DictConfig): Hydra configuration object containing filter parameters.

    Returns:
        List[ray.ObjectRef]: List of Ray actor handles for filter objects to apply.
    """
    filters = []

    if config.filter.use_general_filter:
        filters.append(GeneralFilter.remote())

    if config.filter.task_specific.enabled:
        target_task = config.filter.task_specific.target_task

        if config.filter.task_specific.use_trak:
            if config.private:
                filters.append(
                    DPTRAKFilter.remote(
                        threshold=config.filter.task_specific.trak_threshold,
                        target_task=target_task,
                        epsilon=config.privacy_budget.epsilon,
                        delta=config.privacy_budget.delta,
                    )
                )
            else:
                filters.append(
                    TRAKFilter.remote(
                        threshold=config.filter.task_specific.trak_threshold,
                        target_task=target_task,
                    )
                )

        if config.filter.task_specific.use_clip_score:
            if config.private:
                filters.append(
                    DPCLIPScoreFilter.remote(
                        threshold=config.filter.task_specific.clip_score_threshold,
                        target_task=target_task,
                        epsilon=config.privacy_budget.epsilon,
                        delta=config.privacy_budget.delta,
                    )
                )
            else:
                filters.append(
                    CLIPScoreFilter.remote(
                        threshold=config.filter.task_specific.clip_score_threshold,
                        target_task=target_task,
                    )
                )

        if config.filter.task_specific.use_neg_clip_loss:
            if config.private:
                filters.append(
                    DPNegCLIPLossFilter.remote(
                        threshold=config.filter.task_specific.neg_clip_loss_threshold,
                        target_task=target_task,
                        epsilon=config.privacy_budget.epsilon,
                        delta=config.privacy_budget.delta,
                    )
                )
            else:
                filters.append(
                    NegCLIPLossFilter.remote(
                        threshold=config.filter.task_specific.neg_clip_loss_threshold,
                        target_task=target_task,
                    )
                )

    return filters


def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna optimization.

    Args:
        trial (optuna.Trial): Optuna trial object.

    Returns:
        float: Metric to be optimized (e.g., accuracy).
    """
    with initialize(config_path="../conf"):
        config = compose(config_name="config")

        # Suggest values for hyperparameters
        config.filter.task_specific.trak_threshold = trial.suggest_float(
            "trak_threshold", 0.1, 0.9
        )
        config.filter.task_specific.clip_score_threshold = trial.suggest_float(
            "clip_score_threshold", 0.1, 0.9
        )
        config.filter.task_specific.neg_clip_loss_threshold = (
            trial.suggest_float("neg_clip_loss_threshold", 0.1, 0.9)
        )

        if config.private:
            config.privacy_budget.epsilon = trial.suggest_float(
                "epsilon", 0.1, 10.0, log=True
            )
            config.privacy_budget.delta = trial.suggest_float(
                "delta", 1e-6, 1e-4, log=True
            )

        run_experiment(config)

        # TODO: Return the metric to be optimized
        return 0.0  # Placeholder


def run_optimization(n_trials: int = 100):
    """
    Run hyperparameter optimization using Optuna.

    Args:
        n_trials (int): Number of optimization trials to run.
    """
    study = optuna.create_study(
        direction="maximize", storage=os.environ["OPTUNA"]
    )
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    with initialize(config_path="../conf"):
        config = compose(config_name="config")
        run_experiment(config)

    # Uncomment to run hyperparameter optimization
    # run_optimization()
