import pickle
import numpy as np
from src.data.datagen import load_image_file_paths

with open("result.pickle", "rb") as f:
    test_true, test_pred = pickle.load(f)

test_image_paths = load_image_file_paths("test")

with open("result.csv", "w") as f:
    f.write(",".join(["image_path", "groundtruth", "prediction"]))
    for i in range(len(test_true)):
        f.write(",".join([test_image_paths[i], str(test_true[i]), str(test_pred[i])]))
        f.write("\n")
