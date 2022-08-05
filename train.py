import warnings
warnings.filterwarnings('ignore')

from keras_radam import RAdam
from tensorflow_addons.optimizers import Lookahead
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from models.attention import attention_model
from datagen import DataGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--bs", help="batch size", required=True)
parser.add_argument("--dim", help="dim", required=True)
parser.add_argument("--backbone", help="backbone architecture", required=True)
args = parser.parse_args()

bs = int(args.bs)
dim = (int(args.dim),int(args.dim))

train_images = []
for folder in [os.path.join("train", "fake"), os.path.join("train", "real")]:
	for image in os.listdir(folder):
		train_images.append(os.path.join(folder, image))

train_labels = {}
for image in tqdm(train_images):
    if os.path.dirname(image) == 'real':
        train_labels[image] = 0
    else:
        train_labels[image] = 1

val_images = []
for folder in [os.path.join("test", "fake"), os.path.join("test", "real")]:
	for image in os.listdir(folder):
		val_images.append(os.path.join(folder, image))

val_labels = {}
for image in tqdm(val_images):
    if os.path.dirname(image) == 'real':
        val_labels[image] = 0
    else:
        val_labels[image] = 1

train_gen = DataGenerator(train_images, train_labels, batch_size=bs, dim=dim, type_gen='train')
val_gen = DataGenerator(val_images, val_labels, batch_size=bs, dim=dim, type_gen='test')

X, Y = train_gen[0]

model = attention_model(1, backbone=args.backbone, shape=(dim[0], dim[1], 3))

optimizer = Lookahead(RAdam())
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

validate_freq = 1
start_epoch = 0
filepath = os.path.join("weights", "weight-{epoch:02d}-{accuracy:.2f}-{val_accuracy:.2f}-{val_loss:.5f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, period=validate_freq)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.95, patience=2, verbose=1, mode='auto')

callbacks_list = [checkpoint, reduce_lr]

# Train model on dataset
print("FITTING")
model.fit_generator(generator=train_gen,
                    validation_data=val_gen,
                    epochs=90,
                    verbose=1,
                    callbacks=callbacks_list,
                    initial_epoch=start_epoch,
                    validation_freq=validate_freq,
                    max_queue_size=20,
                    workers = 8,
                    )
