"""
PhysioNet Dataset Downloader
==============================
Auto-downloads MIT-BIH Arrhythmia, MIT-BIH AFDB, and CinC 2017 datasets.
"""

import os
import logging
from pathlib import Path
from typing import List

import wfdb

logger = logging.getLogger(__name__)


# Dataset registry: (name, physionet_db_name, local_dir_name)
DATASETS = {
    "mitdb": {
        "db_name": "mitdb",
        "description": "MIT-BIH Arrhythmia Database (48 records, 360Hz, 2-channel)",
        "records": 48,
    },
    "afdb": {
        "db_name": "afdb",
        "description": "MIT-BIH AF Database (25 records, 250Hz, 10-hour)",
        "records": 25,
    },
    "ltafdb": {
        "db_name": "ltafdb",
        "description": "Long-Term AF Database (84 records, 128Hz, 24-hour)",
        "records": 84,
    },
}


def download_dataset(name: str, data_dir: str, overwrite: bool = False) -> Path:
    """
    Download a PhysioNet dataset.

    Args:
        name: Dataset key from DATASETS registry
        data_dir: Root data directory
        overwrite: Re-download if already exists

    Returns:
        Path to the downloaded dataset directory
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    info = DATASETS[name]
    target_dir = os.path.join(data_dir, name)

    if os.path.exists(target_dir) and not overwrite:
        n_files = len([f for f in os.listdir(target_dir) if f.endswith(".dat")])
        if n_files > 0:
            logger.info(f"Dataset '{name}' already exists at {target_dir} ({n_files} .dat files). Skipping.")
            return Path(target_dir)

    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Downloading {info['description']}...")

    try:
        wfdb.dl_database(info["db_name"], dl_dir=target_dir)
        logger.info(f"Successfully downloaded '{name}' to {target_dir}")
    except Exception as e:
        logger.error(f"Failed to download '{name}': {e}")
        raise

    return Path(target_dir)


def download_all(data_dir: str, datasets: List[str] = None) -> dict:
    """Download all or specified datasets."""
    if datasets is None:
        datasets = ["mitdb", "afdb"]  # Default: the two we need most

    results = {}
    for name in datasets:
        try:
            results[name] = download_dataset(name, data_dir)
        except Exception as e:
            logger.error(f"Skipping {name}: {e}")
            results[name] = None

    return results


def get_record_list(dataset_dir: str) -> List[str]:
    """Get list of record names from a dataset directory."""
    dat_files = sorted([
        f.replace(".dat", "")
        for f in os.listdir(dataset_dir)
        if f.endswith(".dat")
    ])
    return dat_files
