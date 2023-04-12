import tensorflow as tf

from src.metrics.eer import EqualErrorRate

# Register the new metric with TensorFlow
tf.keras.utils.get_custom_objects()['equal_error_rate'] = EqualErrorRate
