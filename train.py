import warnings
warnings.filterwarnings('ignore')

from tensorflow_addons.optimizers import Lookahead, RectifiedAdam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
import os
from tqdm import tqdm
import argparse

from models.attention import attention_model
from datagen import DataGenerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", help="batch size", required=True)
    parser.add_argument("--dim", help="dim", required=True)
    parser.add_argument("--backbone", help="backbone architecture", required=True)
    parser.add_argument("--num-workers", help="number of worker", default=1, required=False)
    
    return parser.parse_args()

def load_dataset_to_generator(data_path, bs, dim, type_gen):
    # generate paths of image
    image_paths = []
    for folder in [os.path.join(data_path, "fake"), os.path.join(data_path, "real")]:
        for path in os.listdir(folder):
            image_paths.append(os.path.join(folder, path))
            
    # generate label
    labels = {}
    for path in tqdm(image_paths):
        labels[path] = int(os.path.dirname(path) == 'real')
    return DataGenerator(image_paths, labels, batch_size=bs, dim=dim, type_gen=type_gen)

# get command line arguments
args = parse_args()
bs = int(args.bs)
dim = (int(args.dim),int(args.dim))
num_workers = int(args.num_workers)

# create data generator
print("---------CREATE DATA GENERATOR---------")
train_gen = load_dataset_to_generator("train", bs, dim, "train")
val_gen = load_dataset_to_generator("test", bs, dim, "test")

# compile model for training
print("---------COMPILE MODEL---------")
model = attention_model(1, backbone=args.backbone, shape=(dim[0], dim[1], 3))
optimizer = Lookahead(RectifiedAdam())
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print("---------INIT CALLBACK---------")
# early stopping callback, stop as long as the validation loss does not decrease anymore
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# CSV Logger callback to save train history
os.makedirs("logs", exist_ok=True)
logger_filepath = os.path.join("logs", "training.log")
csv_logger = CSVLogger(logger_filepath)

# checkpoint callback, save weight every epoch
os.makedirs("weights", exist_ok=True)
checkpoint_filepath = os.path.join("weights", "weight-{epoch:02d}-{accuracy:.2f}-{val_accuracy:.2f}-{val_loss:.5f}.hdf5")
validate_freq = 1
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', verbose=1, save_best_only=False, period=validate_freq)

# reduce learning rate callback, decrease learning rate if val loss does not decrease
# reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.95, patience=2, verbose=1, mode='auto')

# Train model on dataset
print("---------FITTING---------")
start_epoch = 0
callbacks_list = [early_stopping, checkpoint, csv_logger]
model.fit_generator(generator=train_gen,
                    validation_data=val_gen,
                    epochs=90,
                    verbose=1,
                    callbacks=callbacks_list,
                    initial_epoch=start_epoch,
                    validation_freq=validate_freq,
                    max_queue_size=10,
                    workers=num_workers,
                    )
