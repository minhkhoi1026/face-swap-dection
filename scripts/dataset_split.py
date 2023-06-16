import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import argparse

def split(source_path, train_size, sampling_ratio):
    rows = []

    nimg = int(1 / sampling_ratio)

    # Traverse the directory tree
    for root, dirs, files in os.walk(source_path):
        # Loop over the files in the current directory
        files.sort()
        for id,name in enumerate(files):
            if "video" in name and id % nimg != 0:
                continue
            # Check if the file has a .png extension
            if name.endswith(".png"):
                # Print the full path of the file
                file_path = os.path.join(root, name)
                # Get only relative path with source path
                file_path = os.path.relpath(file_path, source_path)
                # real image lie in `real` folder
                label = int("real" in root)
                rows.append((file_path, label))
                  
    df = pd.DataFrame(rows, columns=["filepath", "label"])
    # shuffle to avoid batch contain all 0-labeled or 1-labeled sample
    df = df.sample(frac=1).reset_index(drop=True)

    train, val = train_test_split(df, train_size=train_size)
    
    df.to_csv(os.path.join(source_path, "all_{}.csv".format(int(sampling_ratio*100))), index=None)
    train.to_csv(os.path.join(source_path, "train_split_{}.csv".format(int(sampling_ratio*100))), index=None)
    val.to_csv(os.path.join(source_path, "val_split_{}.csv").format(int(sampling_ratio*100)), index=None)

def split_variant(source_path, train_size, sampling_ratio):
    rows = []
    for imgtype in os.listdir(source_path):
        if "face" in imgtype:
            face_dir = imgtype
        if "landmark" in imgtype:
            landmark_dir = imgtype
    
    nimg = int(1 / sampling_ratio)

    for label in os.listdir(os.path.join(source_path,face_dir)):
        files = os.listdir(os.path.join(source_path,face_dir,label))
        files.sort()
        for id,file in enumerate(files):
            if id % nimg != 0:
                continue
            if file.endswith(".png"):
                file_path = os.path.join(face_dir,label, file)
                name = os.path.splitext(file)[0]+"_landmark.png"
                landmark_path = os.path.join(landmark_dir,label, name)
                if not os.path.exists(os.path.join(source_path,landmark_path)):
                    continue
                rows.append((file_path, landmark_path, int("real" in file_path)))
    
    df = pd.DataFrame(rows, columns=["filepath", "variant", "label"])
    # shuffle to avoid batch contain all 0-labeled or 1-labeled sample
    df = df.sample(frac=1).reset_index(drop=True)

    train, val = train_test_split(df, train_size=train_size)
    
    df.to_csv(os.path.join(source_path, "all_{}.csv".format(int(sampling_ratio*100))), index=None)
    train.to_csv(os.path.join(source_path, "train_split_{}.csv".format(int(sampling_ratio*100))), index=None)
    val.to_csv(os.path.join(source_path, "val_split_{}.csv").format(int(sampling_ratio*100)), index=None)



if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='Specify the source folder path.')
    parser.add_argument('--type', choices=['variant', 'normal'], type=str, default = "normal")
    parser.add_argument('--train-size', type=float, default = 0.85)
    parser.add_argument('--sampling-ratio', type=float, default = 1)
    args = parser.parse_args()

    source_path = args.src
    train_size = args.train_size
    sampling_ratio = args.sampling_ratio
    if args.type == "variant":
        split_variant(source_path, train_size, sampling_ratio)
    else:
        split(source_path, train_size, sampling_ratio)
