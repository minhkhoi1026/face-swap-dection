import argparse
import numpy as np
from sklearn.metrics import roc_curve

from src.data.datagen import DataGenerator
from src.models.attention import attention_model
from src.data.datagen import load_image_file_paths, generate_label_from_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", help="batch size", required=True)
    parser.add_argument("--backbone", help="backbone architecture", required=True)
    parser.add_argument("--dim", help="shape of input image", required=True)
    parser.add_argument("--weight", help="path to model's pretrained weight", required=True)
    return parser.parse_args()

def calculate_err(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    id = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[id], threshold[id]

args = parse_args()

batch_size = int(args.bs)
dim = int(args.dim)

test_image_paths = load_image_file_paths("test")
test_labels = generate_label_from_path(test_image_paths)
test_generator = DataGenerator(test_image_paths, test_labels, batch_size=batch_size, dim=(dim, dim), type_gen='test')

model = attention_model(1, backbone=args.backbone, shape=(dim, dim, 3))
test_preds = model.predict(test_generator)

# threshold, err = calculate_err(test_preds, test_labels)

print(calculate_err(test_preds, test_labels))
