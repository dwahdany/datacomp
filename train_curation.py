import os
import subprocess

import hydra
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("format_ratio", lambda x: f"{float(x):.1f}")  # noqa: E731

TASK_TO_TAR = {
    "fitzpatrick17k": "/datasets/fitzpatrick17k/shards/fitzpatrick17k-train-{000000..000012}.tar",
    "fairvision/DR": "/datasets/fairvision/DR/shards/dr-train-{000000..000005}.tar",
    "fairvision/AMD": "/datasets/fairvision/AMD/shards/amd-train-{000000..000005}.tar",
    "fairvision/Glaucoma": "/datasets/fairvision/Glaucoma/shards/glaucoma-train-{000000..000005}.tar",
    "pcam": "/datasets/pcam/shards/pcam-train-{000000..000262}.tar",
}

OmegaConf.register_new_resolver(
    "get_data_tar",
    lambda task: TASK_TO_TAR.get(task) or ValueError(f"Unknown task: {task}"),
)


@hydra.main(config_path="conf_curation", config_name="sweep")
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
        cfg.curation.indistribution_data_tar,
    ]
    if cfg.dry_run:
        print(" ".join(command))
    else:
        subprocess.run(command, check=True, cwd=os.path.dirname(__file__))


if __name__ == "__main__":
    train()
