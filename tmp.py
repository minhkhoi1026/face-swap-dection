import pickle
import numpy as np
from src.data.datagen import load_image_file_paths

with open("result.pickle", "rb") as f:
    test_true, test_pred = pickle.load(f)

test_image_paths = load_image_file_paths("test")

ids = np.argwhere(np.isnan(test_pred))

for id in ids:
    if "real" not in test_image_paths[id[0]]:
        print(test_image_paths[id[0]])
print(len(ids))
