import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import argparse

def get_id_video(path):
    path = os.path.basename(path)
    return path.split('.')[0].split('_')[0]

def split_variant(source_path, sampling_ratio, hybrid):
    nimg = int(1 / sampling_ratio)

    val_id = ['video023', 'video104', 'video009', 'video008', 'video022', 'video103', 'video102', 'video007', 'video105', '073', '068', '200', '034', '066', '071', '070', '069', '202', '067', '032', '074', '201', '072', '030', '204', '203', '065', '031', '033']
    test_id = ['video033', 'video032', 'video034', 'video127', 'video128', 'video131', 'video031', 'video129', 'video126', '084', '044', '009', '043', '041', '055', '040', '059', '080', '007', '057', '005', '081', '083', '006', '082', '008', '056', '058', '042']
    
    rows = []
    train_rows = []
    val_rows = []
    test_rows = []

    datasets = ['']
    if hybrid:
        datasets = [dir for dir in os.listdir(source_path) if os.path.isdir(os.path.join(source_path,dir))]
    print(datasets)
    for dataset in datasets:
        for imgtype in os.listdir(os.path.join(source_path,dataset)):
            if "face" in imgtype:
                face_dir = imgtype
            if "landmark" in imgtype:
                landmark_dir = imgtype
        for label in os.listdir(os.path.join(source_path,dataset,face_dir)):
            files = os.listdir(os.path.join(source_path,dataset,face_dir,label))
            files.sort()
            for id,file in enumerate(files):
                if id % nimg != 0:
                    continue
                if file.endswith(".png"):
                    file_path = os.path.join(dataset,face_dir,label, file)
                    name = os.path.splitext(file)[0]+"_landmark.png"
                    landmark_path = os.path.join(dataset,landmark_dir,label, name)
                    if not os.path.exists(os.path.join(source_path,landmark_path)):
                        continue
                    source = "original" if "real" in file_path else "deepfaker" if "video" in file else "roop"
                    ilabel = int("real" not in file_path)

                    rows.append((file_path, landmark_path, ilabel, source))
                    if get_id_video(file_path) in val_id:
                        val_rows.append((file_path, landmark_path, ilabel, source))
                    elif get_id_video(file_path) in test_id:
                        test_rows.append((file_path, landmark_path, ilabel, source))
                    else:
                        train_rows.append((file_path, landmark_path, ilabel, source))

    columns=["filepath", "variant", "label", "source"]
    df = pd.DataFrame(rows, columns=columns)
    train = pd.DataFrame(train_rows, columns=columns)
    val = pd.DataFrame(val_rows, columns=columns)
    test = pd.DataFrame(test_rows, columns=columns)
    # shuffle to avoid batch contain all 0-labeled or 1-labeled sample
    df = df.sample(frac=1).reset_index(drop=True)
    train = train.sample(frac=1).reset_index(drop=True)
    val = val.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)
    
    df.to_csv(os.path.join(source_path, "all.csv"), index=None)
    train.to_csv(os.path.join(source_path, "train.csv"), index=None)
    val.to_csv(os.path.join(source_path, "val.csv"), index=None)
    test.to_csv(os.path.join(source_path, "test.csv"), index=None)



if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='Specify the source folder path.')
    # parser.add_argument('--train-size', type=float, default = 0.85)
    parser.add_argument('--sampling-ratio', type=float, default = 1)
    parser.add_argument('--hybrid', action='store_true')
    args = parser.parse_args()

    source_path = args.src
    sampling_ratio = args.sampling_ratio
    hybrid = args.hybrid
    split_variant(source_path, sampling_ratio, hybrid)
