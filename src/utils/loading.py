import random
import numpy as np
import os
import yaml
import pandas as pd


def load_yaml(path):
    with open(path, "rt") as f:
        return yaml.safe_load(f)


def load_image_file_paths(data_path, oversampling=False):
    image_paths = []
    
    dirs = [os.path.join(data_path, "fake"), os.path.join(data_path, "real")]
    
    nsample = max(len(os.listdir(x)) for x in dirs) 

    for folder in dirs:
        tmp_paths = []
        for path in os.listdir(folder):
            tmp_paths.append(os.path.join(folder, path))
        image_paths.extend(tmp_paths)
        
        if oversampling:
            num_add = nsample - len(tmp_paths)
            if num_add == 0: continue
                
            ids = np.arange(len(tmp_paths))
            choices = np.random.choice(ids, num_add)
            image_paths.extend([tmp_paths[id] for id in choices])
            
        print(len(image_paths))

    return image_paths

def generate_label_from_path(image_paths):
    labels = []
    for path in image_paths:
        labels.append(int(os.path.basename(os.path.dirname(path)) == 'real'))
    return labels

def load_image_label(source_path, split_file, oversampling):
    def add_base_path(file_path):
        return os.path.join(source_path, file_path)    
    
    df = pd.read_csv(os.path.join(source_path, split_file))
    
    df["filepath"] = df['filepath'].apply(add_base_path)
    
    if oversampling:
        count_labels = df['label'].value_counts()
        max_count = count_labels.max()
        
        for label in count_labels.index:
            if count_labels[label] < max_count:
                num_rows_to_add = max_count - count_labels[label]
                mask = df['label'] == label
                oversampled_rows = df[mask].sample(n=num_rows_to_add, replace=True)
                df = pd.concat([df, oversampled_rows])
    
    image_paths, landmark_paths, labels = df['filepath'].to_list(), df['landmark'].to_list(), df["label"].to_list()
    
    return image_paths, landmark_paths, labels
