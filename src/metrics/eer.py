import tensorflow as tf
import numpy as np

class EqualErrorRate(tf.keras.metrics.Metric):
    def __init__(self, name='equal_error_rate', thresholds=None, **kwargs):
        super(EqualErrorRate, self).__init__(name=name, **kwargs)
        self.total_batch = self.add_weight(name='total_batch', initializer='zeros')
        self.total_fars = self.add_weight(name='total_fars', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute false acceptance and false rejection rates for each threshold
        thresholds = tf.range(0.0, 1.01, 0.01)
        fars = tf.TensorArray(dtype=tf.float32, size=len(y_true), dynamic_size=True)
        frrs = tf.TensorArray(dtype=tf.float32, size=len(y_true), dynamic_size=True)
        for threshold in thresholds:
            y_pred_binary = tf.cast(tf.math.greater_equal(y_pred, threshold), tf.int32)
            
            fars = fars.write(
                1,
                tf.reduce_mean(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred_binary, 1)), tf.float32), axis=0)
            )

            frrs = frrs.write(
                1,
                tf.reduce_mean(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_binary, 0)), tf.float32), axis=0)
            )

        # Choose threshold that minimizes the difference between FAR and FRR
        min_diff = tf.abs(tf.stack(fars.read(0)) - tf.stack(frrs.read(0)))
        best_index = tf.argmin(min_diff)
        best_threshold = thresholds[best_index]

        # Update weights for best threshold
        y_pred_binary = tf.cast(tf.math.greater_equal(y_pred, best_threshold), tf.int32)
        far = tf.reduce_mean(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred_binary, 1)), tf.float32), axis=0)
        # frr = tf.reduce_mean(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_binary, 0)), tf.float32), axis=0)

        self.total_batch.assign_add(1)
        self.total_fars.assign_add(tf.squeeze(far))
        # self.total_frrs.assign_add(tf.squeeze(frr))

    def result(self):
        # Compute equal error rate for best threshold
        return self.total_fars / self.total_batch

    def reset_states(self):
        self.total_batch.assign(0)
        self.total_fars.assign(0)
