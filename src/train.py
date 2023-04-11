import warnings
warnings.filterwarnings('ignore')

import datetime
import tensorflow.keras as keras
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import wandb

from src.models.attention import attention_model
from src.data.datagen import load_image_file_paths, generate_label_from_path, DataGenerator
from src.callback import CALLBACK_REGISTRY
from src.utils.opt import Opts
from src.utils.loading import load_gpu

def train(config):
    # init new wandb run
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config['global']['run_name']}-{time_str}"
    
    wandb.init(project=config["global"]["project_name"],
				name=run_name,
				entity=config["global"]["username"],
				config=config)
    
    # Check for GPU availability
    load_gpu()

    # create data generator
    print("---------CREATE DATA GENERATOR---------")
    input_shape = (config["model"]["input_size"], config["model"]["input_size"])

    # split train and validation
    image_paths = load_image_file_paths(data_path=config["dataset"]["data_path"],
                                        oversampling=config["dataset"]["oversampling"])
    labels = generate_label_from_path(image_paths)

    X_train, X_val, y_train, y_val = train_test_split(image_paths, 
                                                      labels, 
                                                      train_size=config["dataset"]["train_split"],
                                                      random_state=42
                                                      )

    train_gen = DataGenerator(list_IDs=X_train, 
								labels=y_train, 
								num_classes=config["model"]["num_classes"],
								batch_size=config["data_generator"]["train"]["batch_size"], 
								dim=input_shape, 
								type_gen=config["data_generator"]["train"]["type_gen"]
								)
    val_gen = DataGenerator(list_IDs=X_val, 
							labels=y_val, 
							num_classes=config["model"]["num_classes"],
							batch_size=config["data_generator"]["val"]["batch_size"], 
							dim=input_shape, 
							type_gen=config["data_generator"]["val"]["type_gen"], 
							)

    # compile model for training
    print("---------COMPILE MODEL---------")
    model = attention_model(num_classes=config["model"]["num_classes"], 
                            backbone=config["model"]["backbone"], 
                            shape=(*input_shape, 3))
    optimizer = SGD(**config["optimizer"])
    # optimizer = Lookahead(RectifiedAdam())
    model.compile(optimizer=optimizer, loss=config["model"]["loss"], metrics=config["model"]["metrics"])

    print("---------INIT CALLBACK---------")
    callbacks = [
        CALLBACK_REGISTRY.get(mcfg["name"])(**mcfg["params"])
        for mcfg in config["callbacks"]
    ]

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.95, patience=2, verbose=1, mode='auto')

    # Train model on dataset
    print("---------FITTING---------")
    model.fit(train_gen,
            validation_data=val_gen,
            epochs=config["train"]["num_epoch"],
            verbose=1,
            callbacks=callbacks,
            initial_epoch=config["train"]["start_epoch"],
            validation_freq=config["train"]["validate_freq"],
            max_queue_size=10,
            workers=config["train"]["num_worker"],
            shuffle=True
            )
    
if __name__ == "__main__":
    cfg = Opts(cfg="configs/template.yml").parse_args()
    train(cfg)
