import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

def split(source_path, train_size):
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
    
    df = pd.DataFrame(rows, columns=["filepath", "landmarkpath", "label"])
    # shuffle to avoid batch contain all 0-labeled or 1-labeled sample
    df = df.sample(frac=1).reset_index(drop=True)

    train, val = train_test_split(df, train_size=train_size)
    
    df.to_csv(os.path.join(source_path, "all.csv"), index=None)
    train.to_csv(os.path.join(source_path, "train_split.csv"), index=None)
    val.to_csv(os.path.join(source_path, "val_split.csv"), index=None)

if __name__ == "__main__":
    source_path = sys.argv[1]
    train_size = float(sys.argv[2])
    split(source_path, train_size)
