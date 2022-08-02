import argparse
import os
import tqdm
import numpy as np
from sklearn.metrics import roc_curve

from datagen import DataGenerator
from attention import attention_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", help="batch size", required=True)
    parser.add_argument("--backbone", help="backbone architecture", required=True)
    parser.add_argument("--dim", help="shape of input image", required=True)
    return parser.parse_args()

def load_image_file_paths(root_folder):
    image_paths = []
    for folder in [os.path.join(root_folder, "real"), os.path.join(root_folder, "fake")]:
        for image in os.listdir(folder):
            image_paths.append(os.path.join(folder, image))
    return image_paths

def generate_label_from_path(image_paths):
    labels = {}
    for img_path in tqdm(image_paths):
        if os.path.dirname(img_path) == 'real':
            labels[img_path] = 0
        else:
            labels[img_path] = 1
            
def calculate_err(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    id = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[id], threshold[id]

args = parse_args()

batch_size = int(args.bs)
dim = args.dim

test_image_paths = load_image_file_paths("test")
test_labels = generate_label_from_path(test_image_paths)
test_generator = DataGenerator(test_image_paths, test_labels, batch_size=batch_size, dim=dim, type_gen='test')

model = attention_model(1, backbone=args.backbone, shape=(dim[0], dim[1], 3))
test_preds = model.predict(test_generator)

threshold, err = calculate_err(test_preds, test_labels)
