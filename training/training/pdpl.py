import glob
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def _read_parquet(file):
    return pd.read_parquet(file)

def get_ood_scores(args, scores_zarr):
    metadata_dir = "/datasets/datacomp/metadata"
    parquet_files = glob.glob(os.path.join(metadata_dir, "*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    # Process parquet files in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        # Map maintains order of completion based on input order
        dfs = list(tqdm(
            executor.map(_read_parquet, parquet_files),
            desc="Loading parquet files",
            total=len(parquet_files)
        ))
    
    metadata_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(metadata_df):,} total samples")
    
    uid_path = "/datasets/datacomp/present_uids.pkl"
    if os.path.exists(uid_path):
        with open(uid_path, "rb") as f:
            uids = pickle.load(f)
    else:
        raise ValueError("UIDs not found")
        
    download_mask = metadata_df.uid.isin(uids).to_numpy()
    download_idx = np.where(download_mask)[0]
    
    if args.curation_method == "random":
        ood_scores = np.zeros(len(download_idx))
    else:
        ood_scores = scores_zarr[args.curation_method][args.curation_task][
            "ood_scores"
        ][download_idx]
        
    ood_uids = metadata_df.uid[download_idx]
    return ood_scores, ood_uids
