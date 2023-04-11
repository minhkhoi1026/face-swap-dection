import yaml
import tensorflow as tf

def load_yaml(path):
    with open(path, "rt") as f:
        return yaml.safe_load(f)

def load_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to use only the first GPU
        try:
            tf.config.experimental.set_memory_growth(gpus[0], enable=True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            print(e)
    else:
        print("[WARNING] --------------No GPU found!-------------")
