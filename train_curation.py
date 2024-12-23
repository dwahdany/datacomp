import os
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from conf_curation import resolvers  # noqa: F401

OmegaConf.register_new_resolver("format_ratio", lambda x: f"{float(x):.1f}")  # noqa: E731


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
    output_path = (
        Path(cfg.base_output_dir)
        / f"datacomp_v{cfg.seed}"
        / "small_scale"
        / "checkpoints"
        / "epoch_5.pt"
    )
    print(output_path)
    if output_path.exists():
        print(f"Output path {output_path} already exists, skipping")
        return
    if cfg.dry_run:
        print(" ".join(command))
    else:
        subprocess.run(command, check=True, cwd=os.path.dirname(__file__))


if __name__ == "__main__":
    train()
