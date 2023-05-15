"""
WARNING: you must download the authentication file from https://www.kaggle.com/settings
and move it to ~/.kaggle directory
"""
import argparse
import os

from kaggle.api.kaggle_api_extended import KaggleApi

CASIA_FASD_URL = "minhkhoi1026/casiafasd"
DEEPFAKERAPP_URL = "huynhngotrungtruc/fsd-deepfakerapp"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", 
                        choices = ["casia-fasd", "fsd-deepfakerapp"], 
                        help="specify dataset name for download", 
                        required=True)
    return parser.parse_args()

def download_dataset(dataset_name):
    url = None
    if dataset_name == "casia-fasd":
        url = CASIA_FASD_URL
    elif dataset_name == "fsd-deepfakerapp":
        url = DEEPFAKERAPP_URL
    else:
        return
    
    api = KaggleApi()
    api.authenticate()
    os.makedirs(dataset_name, exist_ok=True)
    api.dataset_download_files(url, path=dataset_name, unzip=True, force=True)

if __name__ == "__main__":
    args = parse_args()
    download_dataset(args.dataset)

