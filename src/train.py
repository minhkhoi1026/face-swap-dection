import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
import os
from datetime import datetime

from src.models.attention import attention_model
from src.data.datagen import load_dataset_to_generator
from src.utils.opt import Opts
from src.utils.loading import load_gpu

def train(config):
    # Check for GPU availability
    load_gpu()
    
    # create data generator
    print("---------CREATE DATA GENERATOR---------")
    datagen_config = config["data_generator"]
    input_shape = (datagen_config["input_size"], datagen_config["input_size"])
    train_gen = load_dataset_to_generator(data_path=datagen_config["train"]["data_path"], 
                                          num_classes=datagen_config["num_classes"],
                                          bs=datagen_config["train"]["batch_size"], 
                                          dim=input_shape, 
                                          type_gen=datagen_config["train"]["type_gen"], 
                                          **datagen_config["train"]["kwargs"])
    val_gen = load_dataset_to_generator(data_path=datagen_config["test"]["data_path"], 
                                          num_classes=datagen_config["num_classes"],
                                          bs=datagen_config["test"]["batch_size"], 
                                          dim=input_shape, 
                                          type_gen=datagen_config["test"]["type_gen"], 
                                          **datagen_config["test"]["kwargs"])

    # compile model for training
    print("---------COMPILE MODEL---------")
    model = attention_model(datagen_config["num_classes"], 
                            backbone=config["model"]["backbone"], 
                            shape=(*input_shape, 3))
    optimizer = SGD(**config["optimizer"])
    # optimizer = Lookahead(RectifiedAdam())
    model.compile(optimizer=optimizer, loss=config["model"]["loss"], metrics=config["model"]["metrics"])

    print("---------INIT CALLBACK---------")
    # early stopping callback, stop as long as the validation loss does not decrease anymore
    # TODO: add callback in config
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    # CSV Logger callback to save train history
    os.makedirs("logs", exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S") 
    log_name = f"training-{time_str}.log"
    logger_filepath = os.path.join("logs", log_name)
    csv_logger = CSVLogger(logger_filepath)

    # checkpoint callback, save weight every epoch
    os.makedirs("weights", exist_ok=True)
    backbone = config["model"]["backbone"]
    checkpoint_filepath = os.path.join("weights", f"weight-{time_str}-{backbone}" + "-{epoch:02d}-{accuracy:.2f}-{val_accuracy:.2f}-{val_loss:.5f}.hdf5")
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, period=config["train"]["validate_freq"])

    # reduce learning rate callback, decrease learning rate if val loss does not decrease
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.95, patience=2, verbose=1, mode='auto')

    # Train model on dataset
    print("---------FITTING---------")
    callbacks_list = [checkpoint, csv_logger]
    model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        epochs=config["train"]["num_epoch"],
                        verbose=1,
                        callbacks=callbacks_list,
                        initial_epoch=config["train"]["start_epoch"],
                        validation_freq=config["train"]["validate_freq"],
                        max_queue_size=10,
                        workers=config["train"]["num_worker"],
                        shuffle=True
                        )
    
if __name__ == "__main__":
    cfg = Opts(cfg="configs/template.yml").parse_args()
    train(cfg)
