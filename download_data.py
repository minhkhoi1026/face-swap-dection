from dotenv import load_dotenv
import argparse
import os

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi

CASIA_FASD_URL = "minhkhoi1026/casiafasd"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", 
                        choices = ["casia-fasd", "oulu-npu", "replay-attack"], 
                        help="specify dataset name for download", 
                        required=True)
    return parser.parse_args()

def download_dataset(dataset_name):
    url = None
    if dataset_name == "casia-fasd":
        url = CASIA_FASD_URL
    else:
        return
    
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('minhkhoi1026/casiafasd', path=os.path.join('dataset', "casia_fasd"), unzip=True)

args = parse_args()
download_dataset(args.dataset)

