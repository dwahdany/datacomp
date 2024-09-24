import os
import subprocess

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    command = [
        "python",
        "train.py",
        "--seed",
        str(cfg.seed),
        "--output_dir",
        cfg.output_dir,
        "--exclude_uids",
        cfg.exclude_uids_path,
    ]

    subprocess.run(command, check=True, cwd=os.path.dirname(__file__))


if __name__ == "__main__":
    train()
