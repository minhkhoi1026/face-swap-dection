from tensorflow.keras.applications.xception import Xception
import tensorflow.keras.backend as K
import tensorflow as tf

from src.models.mobilenet_v3_large import MobileNetV3_Large
from src.models.mobilenet_v3_small import MobileNetV3_Small

class Attention(tf.keras.layers.Layer):
    def __init__(self, input_shape, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.q = self.add_weight(name='kernel_q',
                                shape=(input_shape),
                                initializer='ones')      

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
        })
        return config

    def call(self, X):
        # X (bs, 2, e) with bs=batch_size, e=embed_dim
        # d (bs, 2,) multiple each feature vector with kernel
        d = tf.tensordot(X, self.q, axes=1) 
        # w (bs, 2) softmax vector as weight to fusion
        w = tf.nn.softmax(d, axis=1)
        
        # fusion two vector with w * x => (bs, e)
        return  tf.reduce_sum(X * tf.expand_dims(w, axis=-1), axis=1)

    
def attention_model(num_classes, backbone='MobileNetV3_Small', shape=(256, 256, 3)):
    if backbone == 'Xception':
        stream1 = Xception(include_top=False, weights='imagenet', input_shape=shape, pooling="avg")
        stream2 = Xception(include_top=False, weights='imagenet', input_shape=shape, pooling="avg")
    elif backbone == 'MobileNetV3_Large':
        stream1 = MobileNetV3_Large(shape, num_classes).build()
        stream2 = MobileNetV3_Large(shape, num_classes).build()
    elif backbone == "MobileNetV3_Small": # MobileNetV3_Small
        stream1 = MobileNetV3_Small(shape, num_classes).build()
        stream2 = MobileNetV3_Small(shape, num_classes).build()
    elif backbone == "ViTB8":
        assert shape == (224, 224, 3), "Shape of vision transformer must be 224x224x3"
        import tensorflow_hub as hub
        model_handle = "https://tfhub.dev/sayakpaul/vit_b8_fe/1"
        stream1 = hub.KerasLayer(model_handle, trainable=False, name="stream1")
        stream2 = hub.KerasLayer(model_handle, trainable=False, name="stream2")
    
    # freeze feature extractor
    stream1.trainable = False
    stream2.trainable = False
        
    input1 = tf.keras.layers.Input(shape)
    input2 = tf.keras.layers.Input(shape)
    feat1 = stream1(input1)
    feat2 = stream2(input2)
   
    # rename stream1 and stream2 names to avoid duplicate name in model checkpoint
    stream1._name = "stream1"
    for i in range(len(stream1.weights)):
        stream1.weights[i]._handle_name = 'stream1_' + stream1.weights[i].name
    stream2._name = "stream2"
    for i in range(len(stream2.weights)):
        stream2.weights[i]._handle_name = 'stream2_' + stream2.weights[i].name

    fused_feat = Attention(input_shape=feat1.shape[1])(tf.stack([feat1, feat2], axis=1))
    
    dense1 = tf.keras.layers.Dense(fused_feat.shape[1] // 2, activation="relu", name="dense1")(fused_feat)
    
    dense2 = tf.keras.layers.Dense(dense1.shape[1] // 2, activation="relu", name="dense2")(dense1)

    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(dense2)
    
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)

    return model

if __name__ == "__main__":
    model = attention_model(2, backbone='ViTB8', shape=(224,224,3))
    print(model.summary())
    # from keras.utils import plot_model
    # plot_model(model, 'model.png', show_shapes=True)

