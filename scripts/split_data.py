import os
import argparse
import shutil
import pandas as pd
import json



def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Move files based on prediction and label.')
    parser.add_argument('--result', type=str, help='Path to the result file')
    parser.add_argument('--dest', type=str, help='Path to the destination folder', required=True)
    return parser.parse_args()


def labelification(pred, L = 0.4, R = 0.6):
    if pred < L:
        return 0
    if pred > R:
        return 1
    return -1


def move_files_by_json(json_file, dest):

    # Opening JSON file
    f = open(json_file)
    
    # returns JSON object as a dictionary
    data = json.load(f)

    real2fake = os.path.join(dest,"real2fake")
    fake2real = os.path.join(dest,"fake2real")
    undecided = os.path.join(dest,"undecided")
    os.makedirs(real2fake, exist_ok=True)
    os.makedirs(fake2real, exist_ok=True)
    os.makedirs(undecided, exist_ok=True)

    for pred, label, filepath in data['data']:
        predict = 1 if pred > 0.5 else 0

        if 0.4 < pred < 0.6:
            destination_folder = undecided
            
            # Check if the file exists
            if os.path.isfile(filepath):
                # Copy the file to the destination folder
                shutil.copy(filepath, destination_folder)
                print(f"Copied file {filepath} to {destination_folder}")
            else:
                print(f"File {filepath} does not exist.")

        elif predict != label:
            destination_folder = fake2real if int(predict) == 1 else real2fake

            # Check if the file exists
            if os.path.isfile(filepath):
                # Copy the file to the destination folder
                shutil.copy(filepath, destination_folder)
                print(f"Copied file {filepath} to {destination_folder}")
            else:
                print(f"File {filepath} does not exist.")


def move_files_by_csv(csv_file, dest):

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    real2fake = os.path.join(dest,"real2fake")
    fake2real = os.path.join(dest,"fake2real")
    undecided = os.path.join(dest,"undecided")
    os.makedirs(real2fake, exist_ok=True)
    os.makedirs(fake2real, exist_ok=True)
    os.makedirs(undecided, exist_ok=True)

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        filepath = row['img_path']
        pred = row['pred']
        label = row['target']

        predict = 1 if pred > 0.5 else 0

        if 0.4 < pred < 0.6:
            destination_folder = undecided
            
            # Check if the file exists
            if os.path.isfile(filepath):
                # Copy the file to the destination folder
                shutil.copy(filepath, destination_folder)
                print(f"Copied file {filepath} to {destination_folder}")
            else:
                print(f"File {filepath} does not exist.")

        elif predict != label:
            destination_folder = fake2real if int(predict) == 1 else real2fake

            # Check if the file exists
            if os.path.isfile(filepath):
                # Copy the file to the destination folder
                shutil.copy(filepath, destination_folder)
                print(f"Copied file {filepath} to {destination_folder}")
            else:
                print(f"File {filepath} does not exist.")




args = parse_args()


result_file = args.result
dest = args.dest

ext = os.path.splitext(result_file)[1]

if ext == ".json":
    move_files_by_json(result_file, dest)
elif ext == ".csv":
    move_files_by_csv(result_file, dest)
else:
    print("Invalid result file!")