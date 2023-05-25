import os
import argparse
import shutil
import json



def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Move files based on prediction and label.')
    parser.add_argument('--json', type=str, help='Path to the JSON file')
    parser.add_argument('--dest', type=str, help='Path to the destination folder')
    return parser.parse_args()



def move_files(json_file, dest):

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



args = parse_args()

json_file = args.json
dest = args.dest

move_files(json_file, dest)