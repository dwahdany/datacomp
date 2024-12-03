import os
import subprocess

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf_curation", config_name="config")
def train(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    command = [
        "python",
        "train.py",
        "--seed",
        str(cfg.seed),
        "--output_dir",
        cfg.output_dir,
        "--curation_method",
        cfg.curation.method,
        "--curation_task",
        cfg.curation.task,
        "--curation_ratio",
        str(cfg.curation.ratio),
        "--indistribution_data_tar",
        cfg.indistribution_data_tar,
        "--indistribution_data_num_samples",
        str(cfg.indistribution_data_num_samples),
    ]

    subprocess.run(command, check=True, cwd=os.path.dirname(__file__))


if __name__ == "__main__":
    train()
