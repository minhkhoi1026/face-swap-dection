import os
import argparse
import pandas as pd
import shutil

def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Move files based on prediction and label.')
    parser.add_argument('--csv', type=str, help='Path to the CSV file')
    parser.add_argument('--dest', type=str, help='Path to the destination folder')
    return parser.parse_args()



def move_files(csv_file, dest):

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        filepath = row['filepath']
        predict = row['predict']
        label = row['label']

        real2fake = os.path.join(dest,"real2fake")
        fake2real = os.path.join(dest,"fake2real")
        # Check if predict is different from label
        if predict != label:
            destination_folder = fake2real if int(predict) == 1 else real2fake
            
            # Determine the path to the file to be moved
            file_to_move = filepath

            # Check if the file exists
            if os.path.isfile(file_to_move):
                # Move the file to the destination folder
                shutil.move(file_to_move, destination_folder)
                print(f"Moved file {file_to_move} to {destination_folder}")
            else:
                print(f"File {file_to_move} does not exist.")



args = parse_args()

# Path to the CSV file
csv_file = args.csv

# Path to the destination folder
dest = args.dest

move_files(csv_file, dest)