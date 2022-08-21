import argparse
import numpy as np
import os
from sklearn.metrics import roc_curve, accuracy_score
from tensorflow import argmax
from sklearn.metrics import roc_curve
import pickle

from src.data.datagen import DataGenerator, load_dataset_to_generator, load_image_file_paths, generate_label_from_path
from src.models.attention import attention_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", help="batch size", required=True)
    parser.add_argument("--backbone", help="backbone architecture", required=True)
    parser.add_argument("--dim", help="shape of input image", required=True)
    parser.add_argument("--weight", help="path to model's pretrained weight", required=True)
    parser.add_argument("--num-classes", help="number of output class, can be 1 or 2", required=True)
    return parser.parse_args()

def calculate_err(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    id = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[id], threshold[id]

args = parse_args()

batch_size = int(args.bs)
dim = int(args.dim)
weight_path = args.weight
num_classes = int(args.num_classes)

test_image_paths = load_image_file_paths("test")
test_labels = generate_label_from_path(test_image_paths)
test_generator = DataGenerator(test_image_paths, test_labels, num_classes, batch_size=batch_size, dim=(dim, dim), type_gen='predict')

model = attention_model(num_classes, backbone=args.backbone, shape=(dim, dim, 3))
model.load_weights(weight_path)
    
test_pred = None
if (num_classes == 1):
    test_pred = model.predict(test_generator).flatten()
elif num_classes == 2:
    test_pred = argmax(model.predict(test_generator), axis=1)

test_true = []
for path in test_image_paths:
    test_true.append(int(os.path.basename(os.path.dirname(path)) == 'real'))

with open("result.pickle", "wb") as f:
    pickle.dump([test_true, test_pred], f)

print(calculate_err(test_true, test_pred))
