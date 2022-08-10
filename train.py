import warnings
warnings.filterwarnings('ignore')

from tensorflow_addons.optimizers import Lookahead, RectifiedAdam
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
import os
import argparse
from datetime import datetime

from src.models.attention import attention_model
from src.data.datagen import load_dataset_to_generator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", help="batch size", required=True)
    parser.add_argument("--dim", help="dim", required=True)
    parser.add_argument("--backbone", help="backbone architecture", required=True)
    parser.add_argument("--num-workers", help="number of worker", default=1, required=False)
    
    return parser.parse_args()

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
model = attention_model(2, backbone=args.backbone, shape=(dim[0], dim[1], 3))
optimizer = SGD(learning_rate=0.0001, momentum=0.9)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print("---------INIT CALLBACK---------")
# early stopping callback, stop as long as the validation loss does not decrease anymore
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# CSV Logger callback to save train history
os.makedirs("logs", exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d-%H%M%S") 
log_name = f"training-{time_str}.log"
logger_filepath = os.path.join("logs", log_name)
csv_logger = CSVLogger(logger_filepath)

# checkpoint callback, save weight every epoch
os.makedirs("weights", exist_ok=True)
checkpoint_filepath = os.path.join("weights", f"weight-{args.backbone}" + "-{epoch:02d}-{accuracy:.2f}-{val_accuracy:.2f}-{val_loss:.5f}.hdf5")
validate_freq = 1
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', verbose=1, save_best_only=False, period=validate_freq)

# reduce learning rate callback, decrease learning rate if val loss does not decrease
# reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.95, patience=2, verbose=1, mode='auto')

# Train model on dataset
print("---------FITTING---------")
start_epoch = 0
callbacks_list = [checkpoint, csv_logger]
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
