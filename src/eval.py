import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score
from tensorflow import argmax
from sklearn.metrics import roc_curve
import datetime
import wandb
import pandas as pd

from src.data.datagen import DataGenerator, load_image_file_paths, generate_label_from_path
from src.models.attention import attention_model
from src.utils.loading import load_gpu
from src.utils.opt import Opts

def calculate_err(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    id = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[id], threshold[id]

def eval(config):
    print("-----START-----")
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config['global']['run_name']}-{time_str}"
    
    run = wandb.init(project=config["global"]["project_name"],
				name=run_name,
				entity=config["global"]["username"],
				config=config)
    num_classes = config["model"]["num_classes"]
    load_gpu()

    print("-----CREATE DATA GENERATOR-----")
    input_shape = (config["model"]["input_size"], config["model"]["input_size"])

    test_image_paths = load_image_file_paths(data_path=config["dataset"]["data_path"],
                                             oversampling=config["dataset"]["oversampling"],
                                             shuffle=config["dataset"]["shuffle"])
    test_labels = generate_label_from_path(test_image_paths)
    test_generator = DataGenerator(list_IDs=test_image_paths, 
                                    labels=test_labels, 
                                    num_classes=config["model"]["num_classes"],
                                    batch_size=config["data_generator"]["batch_size"], 
                                    dim=input_shape, 
                                    type_gen=config["data_generator"]["type_gen"], 
                                    shuffle=config["data_generator"]["shuffle"])

    print("-----CREATE MODEL AND LOAD WEIGHT-----")
    model = attention_model(num_classes=config["model"]["num_classes"], 
                            backbone=config["model"]["backbone"], 
                            shape=(*input_shape, 3))
    model.load_weights(config["model"]["weight_path"])
    
    print("-----PREDICT-----")
    test_pred = model.predict(test_generator, verbose=1)

    print("-----SAVE RESULT-----")
    # Create a dataframe with two columns
    if config["global"]["save_pandas"]:
        df = pd.DataFrame({'image_path': test_image_paths, 
                           'groundtruth': [v for v in test_labels], 
                           'predict': [v for v in test_pred]})
        run.log({"eval_result": wandb.Table(dataframe=df)})
    
    if num_classes == 1:
        test_pred = test_pred.flatten()
    elif num_classes == 2:
        test_pred = argmax(test_pred, axis=1)

    print(calculate_err(test_labels, test_pred))

if __name__ == "__main__":
    cfg = Opts(cfg="configs/eval.yml").parse_args()
    eval(cfg)
