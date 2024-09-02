import hydra
from experiment_lib import run_experiment, run_optimization
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig):
    """
    Main entry point for running experiments or optimizations.

    Args:
        config (DictConfig): Hydra configuration object.
    """
    if config.mode == "experiment":
        run_experiment(config)
    elif config.mode == "optimization":
        run_optimization(config.optim.n_trials)
    else:
        raise ValueError(
            f"Invalid mode: {config.mode}. Choose 'experiment' or 'optimization'."
        )


if __name__ == "__main__":
    main()
