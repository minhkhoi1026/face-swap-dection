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

    train, val = train_test_split(df, train_size=train_size)
    
    train.to_csv(os.path.join(source_path, "train_split.csv"), index=None)
    val.to_csv(os.path.join(source_path, "val_split.csv"), index=None)

if __name__ == "__main__":
    source_path = sys.argv[1]
    train_size = float(sys.argv[2])
    split(source_path, train_size)
