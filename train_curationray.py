import multiprocessing
import os
import shutil
import subprocess
from pathlib import Path

import hydra
from cloudpathlib import S3Client
from omegaconf import DictConfig, OmegaConf
import ray
from conf_curation import resolvers  # noqa: F401

# OmegaConf.register_new_resolver("format_ratio", lambda x: f"{float(x):.1f}")  # noqa: E731


def check_if_done(cfg: DictConfig):
    output_path = (
        Path(cfg.base_output_dir)
        / f"datacomp_v{cfg.seed}"
        / "small_scale"
        / "checkpoints"
        / "epoch_5.pt"
    )
    print("local output path", output_path)
    s3_client = S3Client(endpoint_url=cfg.s3_endpoint_url)
    s3_output_path = (
        s3_client.S3Path(cfg.s3_output_dir)
        / f"datacomp_v{cfg.seed}"
        / "small_scale"
        / "checkpoints"
        / "epoch_5.pt"
    )
    print("s3 output path", s3_output_path)
    if output_path.exists():
        print(f"Output path {output_path} already exists, skipping")
        return True
    if s3_output_path.exists():
        print(f"Output path {s3_output_path} already exists on S3, skipping")
        return True
    return False


@hydra.main(config_path="conf_curation", config_name="ray")
def main(cfg: DictConfig):
    del os.environ["RAY_ADDRESS"]
    ray.init(address="auto")
    results = []
    for seed in cfg.seeds:
        for method in cfg.curation.method:
            for task in cfg.curation.task:
                for ratio in cfg.curation.ratios:
                    for id in [True, False]:
                        this_cfg = OmegaConf.to_container(cfg)
                        this_cfg.pop("curation")
                        this_cfg.pop("seeds")
                        this_cfg["curation"] = {}
                        if id:
                            this_cfg["curation"]["indistribution_data_tar"] = (
                                resolvers.TASK_TO_TAR[task]
                            )
                            this_cfg["dir_prefix"] = "curation"
                        else:
                            this_cfg["curation"]["indistribution_data_tar"] = None
                            this_cfg["dir_prefix"] = "curation_no_id"
                        this_cfg["seed"] = seed
                        this_cfg["curation"]["method"] = method
                        this_cfg["curation"]["task"] = task
                        this_cfg["curation"]["ratio"] = ratio
                        this_cfg["base_output_dir"] = (
                            f"/raid/pdpl/small_clip_checkpoints/{this_cfg['dir_prefix']}/{this_cfg['curation']['method']}/{this_cfg['curation']['task']}/ratio_{this_cfg['curation']['ratio']:.1f}"
                        )
                        this_cfg["output_dir"] = (
                            f"{this_cfg['base_output_dir']}/datacomp_v{this_cfg['seed']}/"
                        )
                        this_cfg["s3_output_dir"] = (
                            f"{this_cfg['base_output_dir'].replace('/raid/pdpl', 's3://pdpl')}"
                        )
                        this_cfg = OmegaConf.create(this_cfg)
                        if check_if_done(this_cfg):
                            continue
                        results.append(train.remote(this_cfg))
    ray.get(results)


@ray.remote(num_gpus=1)
def train(cfg: DictConfig):
    try:
        train_no_wrap(cfg)
    except Exception as e:
        print(
            f"Error training {cfg.curation.method} on {cfg.curation.task} with ratio {cfg.curation.ratio}: {e}"
        )


def train_no_wrap(cfg: DictConfig):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
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
        f"{float(cfg.curation.ratio):.1f}",
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
    print("local output path", output_path)
    s3_client = S3Client(endpoint_url=cfg.s3_endpoint_url)
    s3_output_path = (
        s3_client.S3Path(cfg.s3_output_dir)
        / f"datacomp_v{cfg.seed}"
        / "small_scale"
        / "checkpoints"
        / "epoch_5.pt"
    )
    print("s3 output path", s3_output_path)
    if output_path.exists():
        print(f"Output path {output_path} already exists, skipping")
        return
    if s3_output_path.exists():
        print(f"Output path {s3_output_path} already exists on S3, skipping")
        return
    if cfg.dry_run:
        print(" ".join(command))
    else:
        subprocess.run(command, check=True, cwd=os.path.dirname(__file__))
        local_dir = Path(cfg.base_output_dir) / f"datacomp_v{cfg.seed}"
        if local_dir.exists():

            def upload_and_cleanup():
                print(f"Starting upload of directory {local_dir} to S3")
                s3_base_dir = (
                    s3_client.S3Path(cfg.s3_output_dir) / f"datacomp_v{cfg.seed}"
                )

                for local_file in local_dir.rglob("*"):
                    if local_file.is_file():
                        relative_path = local_file.relative_to(local_dir)
                        s3_path = s3_base_dir / relative_path
                        print(f"Uploading {local_file} to {s3_path}")
                        s3_path.write_bytes(local_file.read_bytes())

                shutil.rmtree(local_dir)
                print("Upload complete and local directory cleaned up")

            p = multiprocessing.Process(target=upload_and_cleanup)
            p.start()
            print("Upload process started in background")


if __name__ == "__main__":
    main()
