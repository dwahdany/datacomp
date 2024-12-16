import os
import subprocess

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="glaucoma")
def train(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    command = [
        "python",
        "train.py",
        "--seed",
        str(cfg.seed),
        "--output_dir",
        cfg.output_dir,
    ]

    for k in [
        "indistribution_data_tar",
        "indistribution_data_tar_upsample",
        "indistribution_data_sampling_rate",
        "curation_task",
    ]:
        if k in cfg:
            command.extend([f"--{k}", str(cfg[k])])

    subprocess.run(command, check=True, cwd=os.path.dirname(__file__))


if __name__ == "__main__":
    train()
