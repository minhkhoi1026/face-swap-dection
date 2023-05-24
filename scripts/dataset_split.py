import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

def split(source_path, train_size):
    rows = []
    # Traverse the directory tree
    for root, dirs, files in os.walk(source_path):
        # Loop over the files in the current directory
        for name in files:
            # Check if the file has a .png extension
            if name.endswith(".png"):
                # Print the full path of the file
                file_path = os.path.join(root, name)
                # Get only relative path with source path
                file_path = os.path.relpath(file_path, source_path)
                # real image lie in `real` folder
                label = int(file_path.startswith("real"))
                rows.append((file_path, label))
                  
    df = pd.DataFrame(rows, columns=["filepath", "label"])
    # shuffle to avoid batch contain all 0-labeled or 1-labeled sample
    df = df.sample(frac=1).reset_index(drop=True)

    train, val = train_test_split(df, train_size=train_size)
    
    df.to_csv(os.path.join(source_path, "all.csv"), index=None)
    train.to_csv(os.path.join(source_path, "train_split.csv"), index=None)
    val.to_csv(os.path.join(source_path, "val_split.csv"), index=None)

def split_variant(source_path, train_size):
    rows = []
    for imgtype in os.listdir(source_path):
        if "face" in imgtype:
            face_dir = imgtype
        if "landmark" in imgtype:
            landmark_dir = imgtype
        
    for label in os.listdir(os.path.join(source_path,face_dir)):
        for file in os.listdir(os.path.join(source_path,face_dir,label)):
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
    
    df.to_csv(os.path.join(source_path, "all.csv"), index=None)
    train.to_csv(os.path.join(source_path, "train_split.csv"), index=None)
    val.to_csv(os.path.join(source_path, "val_split.csv"), index=None)

if __name__ == "__main__":
    source_path = sys.argv[1]
    train_size = float(sys.argv[2])
    if sys.argv[3] == "landmark":
        split_variant(source_path, train_size)
    else:
        split(source_path, train_size)
