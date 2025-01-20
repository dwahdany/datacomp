from omegaconf import OmegaConf

TASK_TO_TAR = {
    "fitzpatrick17k": "/datasets/fitzpatrick17k/shards/fitzpatrick17k-train-{000000..000012}.tar",
    "fairvision/DR": "/datasets/fairvision/DR/shards/dr-train-{000000..000005}.tar",
    "fairvision/AMD": "/datasets/fairvision/AMD/shards/amd-train-{000000..000005}.tar",
    "fairvision/Glaucoma": "/datasets/fairvision/Glaucoma/shards/glaucoma-train-{000000..000005}.tar",
    "pcam": "/datasets/pcam/shards/pcam-train-{000000..000262}.tar",
    "food101": "/datasets/food101/shards/food101-train-{000000..000075}.tar",
    "cifar100": "/datasets/cifar100/shards/cifar100-train-{000000..000049}.tar",
    "stl10": "/datasets/stl10/shards/stl10-train-{000000..000004}.tar",
}

OmegaConf.register_new_resolver(
    "get_data_tar",
    lambda task: TASK_TO_TAR[task],
)
