import warnings
warnings.filterwarnings('ignore')

import argparse
from cv2 import imshow
from cv2 import CascadeClassifier
from cv2 import rectangle
import cv2
import numpy as np

from src.utils.retinex import automatedMSRCR
from src.models.attention import attention_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", help="path to model's pretrained weight", required=True)
    parser.add_argument("--cascade-classifier", help="path to cascade classifier", require=True)
    return parser.parse_args()

args = parse_args()
weigth_path = args.weight
cascade_classifier_path = args.cascade_classifier

cap = cv2.VideoCapture(0)

classifier = CascadeClassifier(cascade_classifier_path)
model = attention_model(1, backbone='MobileNetV3', shape=(299, 299, 3))
model.load_weights(weigth_path)
print(model.summary())

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while True:
    ret, frame = cap.read()
    bboxes = classifier.detectMultiScale(frame)

    for box in bboxes:
        x, y, width, height = box
        x2, y2 = int(x + 1.0*width), int(y + 1.2*height)
        x, y = int(x-0.0*width), int(y-0.2*height)

        img = frame[y:y2, x:x2]
        print(type(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))

        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        new_img = np.expand_dims(new_img, -1)
        new_img = automatedMSRCR(new_img, [10, 20, 30])
        new_img = cv2.cvtColor(new_img[:, :, 0], cv2.COLOR_GRAY2RGB)

        preds = model.predict([np.expand_dims(img / 255.0, 0), np.expand_dims(new_img / 255.0, 0)])

        if preds[0][0] > 0.90:
            rectangle(frame, (x, y), (x2, y2), (0,0,255), 1)
        else:
            rectangle(frame, (x, y), (x2, y2), (0,255,0), 1)
        out.write(frame)
    imshow('face detection', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

