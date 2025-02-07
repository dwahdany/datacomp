import ast
import json
import logging
import math
import os
import pickle
import random
import sys
import warnings
from dataclasses import dataclass
from multiprocessing import Value
from typing import Callable, Optional

import braceexpand
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
import zarr
from PIL import Image
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    SubsetRandomSampler,
    get_worker_info,
)
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10, CIFAR100, STL10, Food101
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def get_ood_scores(args, scores_zarr):
    if args.curation_method == "trak":
        scores_zarr = zarr.open("/raid/pdpl/trak_scores.zarr", mode="r")
        if args.indistribution_data_tar is None:
            scores_zarr = scores_zarr["raw"][args.curation_task.lower()]
        else:
            scores_zarr = scores_zarr[args.curation_task.lower()][
                args.curation_task.lower()
            ]
        scores = scores_zarr["ood_scores"][:]
        uids = scores_zarr["ood_uids"][:].astype("U32")
        return scores, uids

    datacomp = zarr.open("/datasets/datacomp/metadata.zarr", mode="r")
    ood_uids = datacomp["uid"][:].astype("U32")

    uid_path = "/datasets/datacomp/present_uids.pkl"
    if os.path.exists(uid_path):
        with open(uid_path, "rb") as f:
            uids = np.asarray(pickle.load(f), dtype="U32")
    else:
        raise ValueError("UIDs not found")

    download_mask = np.isin(ood_uids, uids)
    download_idx = np.where(download_mask)[0]

    if args.curation_method == "random":
        ood_scores = np.zeros(len(download_idx))
    else:
        ood_scores = scores_zarr[args.curation_method][args.curation_task][
            "ood_scores"
        ][download_idx]

    ood_uids = ood_uids[download_idx]
    return ood_scores, ood_uids


class CsvDataset(Dataset):
    def __init__(
        self,
        input_filename,
        transforms,
        img_key,
        caption_key,
        sep="\t",
        tokenizer=None,
    ):
        logging.debug(f"Loading csv data from {input_filename}.")
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(
            self.sampler, DistributedSampler
        ):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split("::")
        assert len(weights) == len(urllist), (
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        )
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum(
            [int(sizes[os.path.basename(shard)]) for shard in shards_list]
        )
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, "r").read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset

        dataset = ImageNetV2Dataset(
            location=args.imagenet_v2, transform=preprocess_val
        )
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype("int")
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample or "label" in sample
    has_image = (
        "png" in sample
        or "jpg" in sample
        or "jpeg" in sample
        or "webp" in sample
    )
    # print(sample.keys(), has_caption, has_image)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                # print("Valid sample")
                yield current_sample
            else:
                print("Invalid sample")
            current_sample = dict(
                __key__=prefix, __url__=filesample["__url__"]
            )
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), (
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
            )
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(
                    url=self.rng.choices(self.urls, weights=self.weights, k=1)[
                        0
                    ]
                )


def get_wds_dataset(
    args,
    preprocess_img,
    is_train,
    epoch=0,
    floor=False,
    tokenizer=None,
    ood_filter: Optional[Callable] = None,
    id_filter: Optional[Callable] = None,
    debug: bool = False,
):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    "Currently, the number of dataset samples must be specified for the training dataset. "
                    "Please specify it via `--train-num-samples` if no dataset length info is present."
                )
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(
        epoch=epoch
    )  # create a shared epoch store to sync epoch to dataloader worker proc

    if resampled:
        dataset = ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )
    else:
        assert args.train_data_upsampling_factors is None, (
            "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
        )
        dataset = wds.SimpleShardList(input_shards)

    # dataset = wds.DataPipeline(dataset, wds.decode("pilrgb"))

    pipelines = [[dataset]]

    if args.indistribution_data_tar is not None:
        task = args.curation_task

        ds_class = {
            "cifar10": CIFAR10,
            "cifar100": CIFAR100,
            "food101": Food101,
            "stl10": STL10,
        }
        if task.lower() in ds_class.keys():
            classnames = ds_class[task.lower()](root="/datasets").classes
        else:
            with open(
                "/git/CLIP_benchmark/clip_benchmark/datasets/en_classnames.json"
            ) as f:
                classnames = json.load(f)[task.lower()]

        with open(
            "/git/CLIP_benchmark/clip_benchmark/datasets/en_zeroshot_classification_templates.json"
        ) as f:
            template = json.load(f)[task.lower()][0]

        filled_templates = [template.replace("{c}", c) for c in classnames]

        def label_map(label):
            return filled_templates[int(label)]

        # class_names = {
        #     "diabetic_retinopathy": [
        #         "no diabetic retinopathy",
        #         "mild diabetic retinopathy",
        #         "moderate diabetic retinopathy",
        #         "severe diabetic retinopathy",
        #         "proliferative diabetic retinopathy",
        #     ],
        #     "fairvision/amd": [
        #         "no age-related macular degeneration",
        #         "early age-related macular degeneration",
        #         "intermediate age-related macular degeneration",
        #         "late age-related macular degeneration",
        #     ],
        #     "fairvision/dr": [
        #         "no vision threatening diabetic retinopathy",
        #         "vision threatening diabetic retinopathy",
        #     ],
        #     "fairvision/glaucoma": ["without glaucoma", "with glaucoma"],
        #     "fitzpatrick17k": [
        #         "benign lesion",
        #         "malignant lesion",
        #         "non-neoplastic condition",
        #     ],
        #     "pcam": [
        #         "a histopathology slide showing lymph node",
        #         "histopathology image of lymph node containing metastatic tumor tissue",
        #     ],
        # }
        # templates = {
        #     "diabetic_retinopathy": ["a retinal image with {c}."],
        #     "fairvision/amd": ["a retinal image with {c}."],
        #     "fairvision/dr": ["a retinal image with {c}."],
        #     "fairvision/glaucoma": ["a retinal image {c}."],
        #     "fitzpatrick17k": ["a skin image showing a {c}."],
        #     "pcam": ["{c}."],
        # }

        # def label_map(label):
        #     return templates[task][0].format(c=class_names[task][int(label)])

        id_zarr = zarr.open("/raid/pdpl/id_downstream_idx.zarr", mode="r")
        id_uids = set(
            id_zarr[args.curation_task.lower()]["id_indices"][:].tolist()
        )
        indistribution_data_num_samples = len(id_uids)

        p_id = args.indistribution_data_sampling_rate
        if p_id is None:
            p_id = (
                args.indistribution_data_tar_upsample
                * indistribution_data_num_samples
                / args.train_num_samples
            )

        print("Indistribution data mixing. Parameters:")
        print(f"Task: {task}")
        print(f"Templates: {template}")
        print(f"Class names: {classnames}")
        # print(f"Templates: {templates[task]}")
        # print(f"Class names: {class_names[task]}")
        print(f"Tar file: {args.indistribution_data_tar}")
        print(f"Upsampling factor: {args.indistribution_data_tar_upsample}")
        print(f"Sampling rate: {p_id}")
        probs = [1 - p_id, p_id]
        print(f"Probs: {probs}")
        id_dataset = ResampledShards2(
            args.indistribution_data_tar, deterministic=True
        )
        pipelines.append([id_dataset])

    ood_filters = []
    id_filters = []
    if ood_filter is not None:
        ood_filters.append(ood_filter)
    if id_filter is not None:
        id_filters.append(id_filter)

    if "exclude_uids" in args:
        if args.exclude_uids is not None:
            exclude_uids = None

            def uid_to_str(uid):
                return format(int(uid[0]) << 64 | uid[1], "032x")

            exclude_uids = np.load(args.exclude_uids)
            exclude_uids = [uid_to_str(uid) for uid in exclude_uids]

            def filter_uids(sample):
                return (
                    json.loads(sample["json"].decode("utf-8"))["uid"]
                    not in exclude_uids
                )

            ood_filters.append(filter_uids)

            print(f"Created exclusion filter for {len(exclude_uids)} UIDs")

    if args.curation_method is not None:
        scores_zarr = zarr.open(
            "/raid/pdpl/nearest_neighbor_scores.zarr", mode="r"
        )
        id_zarr = zarr.open("/raid/pdpl/id_downstream_idx.zarr", mode="r")
        print(f"Enabling curation method: {args.curation_method}")
        ood_scores, ood_uids = get_ood_scores(args, scores_zarr)
        print(f"Loaded {len(ood_scores):,} OOD scores")
        try:
            id_uids = np.array(
                id_zarr[args.curation_task.lower()]["id_indices"]
            )
        except KeyError as e:
            print(f"No id indices found for {args.curation_task}")
            print("ID zarr structure:")

            def print_zarr_structure(group, indent=""):
                for key, value in group.items():
                    if isinstance(value, zarr.Group):
                        print(f"{indent}{key}/")
                        print_zarr_structure(value, indent + "  ")
                    else:
                        print(
                            f"{indent}{key}: shape={value.shape}, dtype={value.dtype}"
                        )

            print_zarr_structure(id_zarr)
            raise e

        print(f"Aiming to keep {args.curation_ratio:.2%} of the data")
        if args.curation_method == "random":
            if args.curation_ratio == 1.0:
                selected_ood_idx = set(ood_uids)
            else:
                rng = np.random.default_rng(args.seed)
                selected_ood_idx = set(
                    rng.choice(
                        ood_uids,
                        size=int(len(ood_scores) * args.curation_ratio),
                        replace=False,
                    )
                )
            if args.indistribution_data_tar is not None:
                if args.curation_ratio == 1.0:
                    selected_id_idx = set(id_uids)
                else:
                    selected_id_idx = set(
                        rng.choice(
                            id_uids,
                            size=int(len(id_uids) * args.curation_ratio),
                            replace=False,
                        )
                    )
        else:
            if args.indistribution_data_tar is not None:
                if args.curation_method == "trak":
                    trak_scores_zarr = zarr.open(
                        "/raid/pdpl/trak_scores.zarr", mode="r"
                    )
                    id_scores = trak_scores_zarr[args.curation_task.lower()][
                        args.curation_task.lower()
                    ]["id_scores"][:]
                    id_uids = trak_scores_zarr[args.curation_task.lower()][
                        args.curation_task.lower()
                    ]["id_uids"][:]
                else:
                    id_scores = scores_zarr[args.curation_method][
                        args.curation_task
                    ]["id_scores"][:]
                all_scores = np.concatenate([ood_scores, id_scores])
            else:
                all_scores = ood_scores
            cutoff_value = np.percentile(
                all_scores, (1 - args.curation_ratio) * 100
            )
            selected_ood_idx = set(ood_uids[ood_scores >= cutoff_value])
            if args.indistribution_data_tar is not None:
                selected_id_idx = set(
                    id_uids[np.where(id_scores >= cutoff_value)[0]]
                )
            print(f"Cutoff value: {cutoff_value}")
        if args.indistribution_data_tar is not None:
            selected_id_str = {f"{uid:08d}" for uid in selected_id_idx}
            print(
                f"Number of OOD samples: {len(selected_ood_idx), type(selected_ood_idx)}"
            )
            print(
                f"Number of ID samples: {len(selected_id_idx), type(selected_id_idx)}"
            )
            print(
                f"Keeping {(len(selected_id_idx) + len(selected_ood_idx)) / (len(ood_uids) + len(id_uids)):.2%} of the data ({len(selected_id_idx) / len(id_uids):.2%} id / {len(selected_ood_idx) / len(ood_uids):.2%} ood)"
            )
            p_id = len(selected_id_idx) / (
                len(selected_id_idx) + len(selected_ood_idx)
            )
            print(f"Indistribution data sampling rate: {p_id}")
            if args.indistribution_data_sampling_rate is not None:
                warnings.warn(
                    "Indistribution data sampling rate already specified, overriding with curation method."
                )
                print(f"Old value: {args.indistribution_data_sampling_rate}")
                print(f"New value: {p_id}")
            else:
                args.indistribution_data_sampling_rate = p_id

            def id_filter(sample):
                return sample["__key__"] in selected_id_str

            id_filters.append(id_filter)
        else:
            print(
                f"Number of OOD samples: {len(selected_ood_idx), type(selected_ood_idx)}"
            )
            print(
                f"Keeping {len(selected_ood_idx) / len(ood_uids):.2%} of the data. No ID data."
            )

        def ood_filter(sample):
            uid = json.loads(sample["json"].decode("utf-8"))["uid"]
            return uid in selected_ood_idx

        ood_filters.append(ood_filter)

    else:
        id_zarr = zarr.open("/raid/pdpl/id_downstream_idx.zarr", mode="r")
        id_uids = set(id_zarr[args.curation_task]["id_indices"])

        def id_filter(sample):
            return sample["__key__"] in id_uids

        id_filters.append(id_filter)

    for i, pipeline in enumerate(pipelines):
        # at this point we have an iterator over all the shards
        if is_train:
            if not resampled:
                pipeline.extend(
                    [
                        detshuffle2(
                            bufsize=_SHARD_SHUFFLE_SIZE,
                            initial=_SHARD_SHUFFLE_INITIAL,
                            seed=args.seed,
                            epoch=shared_epoch,
                        ),
                        wds.split_by_node,
                        wds.split_by_worker,
                    ]
                )
            pipeline.extend(
                [
                    # at this point, we have an iterator over the shards assigned to each worker at each node
                    tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
                ]
            )
            if i == 0:
                for filter in ood_filters:
                    pipeline.extend([wds.select(filter)])
            elif i == 1:
                for filter in id_filters:
                    pipeline.extend([wds.select(filter)])
            pipeline.extend(
                [
                    wds.shuffle(
                        bufsize=_SAMPLE_SHUFFLE_SIZE,
                        initial=_SAMPLE_SHUFFLE_INITIAL,
                    ),
                ]
            )
        else:
            pipeline.extend(
                [
                    wds.split_by_worker,
                    # at this point, we have an iterator over the shards assigned to each worker
                    wds.tarfile_to_samples(handler=log_and_continue),
                ]
            )
            if i == 0:
                for filter in ood_filters:
                    pipeline.extend([wds.select(filter)])
            elif i == 1:
                for filter in id_filters:
                    pipeline.extend([wds.select(filter)])

            pipeline.extend([wds.select(filter_no_caption_or_no_image)])
    # if ood_filter is not None:
    #     pipelines[0].extend([wds.select(ood_filter)])
    # if id_filter is not None:
    #     pipelines[1].extend([wds.select(id_filter)])
    if debug:
        return pipelines
    if len(pipelines) > 1:
        if is_train:
            pipeline = [
                wds.RandomMix(
                    [wds.DataPipeline(*pipeline) for pipeline in pipelines],
                    probs=[1 - p_id, p_id],
                    longest=False,
                )
            ]
        else:
            raise ValueError("Eval pipeline should not mix ID and OOD data")
    pipeline.extend(
        [
            wds.decode("pilrgb", handler=log_and_continue),
            wds.map(
                lambda sample: {
                    **sample,
                    "txt": label_map(sample["label"])
                    if "txt" not in sample
                    else sample["txt"],
                }
            ),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map_dict(
                image=preprocess_img, text=lambda text: tokenizer(text)[0]
            ),
            wds.to_tuple("image", "text"),
            wds.batched(args.batch_size, partial=not is_train),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, (
                "number of shards must be >= total workers"
            )
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(
            num_batches / num_workers
        )  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(
            num_worker_batches
        )  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
    )
    num_samples = len(dataset)
    sampler = (
        DistributedSampler(dataset) if args.distributed and is_train else None
    )
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):
    def __init__(
        self,
        transform=None,
        image_size=(224, 224),
        caption="Dummy caption",
        dataset_size=100,
        tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new("RGB", image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(
    args, preprocess_fn, is_train, epoch=0, tokenizer=None
):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn,
        image_size=image_size,
        dataset_size=args.train_num_samples,
        tokenizer=tokenizer,
    )
    num_samples = len(dataset)
    sampler = (
        DistributedSampler(dataset) if args.distributed and is_train else None
    )
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


# def get_indistribution_data(args, preprocess_fns, dataset_name):
#     dataset = build_dataset(
#         dataset_name=dataset_name,
#         root=args.indistribution_data_dir,
#         transform=preprocess_fns,
#         split="train",
#         download=True,
#     )
#     sampler = DistributedSampler(dataset) if args.distributed and is_train else None
#     is_train = True
#     shuffle = is_train and sampler is None
#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=shuffle,
#         num_workers=args.workers,
#         pin_memory=True,
#         sampler=sampler,
#         drop_last=is_train,
#     )
#     return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split(".")[-1]
        if ext in ["csv", "tsv"]:
            return get_csv_dataset
        elif ext in ["tar"]:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}."
            )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(
    args,
    preprocess_fns,
    epoch=0,
    tokenizer=None,
    id_filter: Optional[Callable] = None,
    ood_filter: Optional[Callable] = None,
    debug: bool = False,
):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args,
            preprocess_train,
            is_train=True,
            epoch=epoch,
            tokenizer=tokenizer,
            id_filter=id_filter,
            ood_filter=ood_filter,
            debug=debug,
        )

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args,
            preprocess_val,
            is_train=False,
            tokenizer=tokenizer,
            id_filter=id_filter,
            ood_filter=ood_filter,
            debug=debug,
        )

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    # if args.indistribution_data is not None:
    #     data["indistribution"] = get_indistribution_data(args, preprocess_train, args.indistribution_data)

    return data
