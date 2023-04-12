from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, \
    Add, Lambda, Layer, Concatenate, Softmax
from tensorflow.keras.applications.xception import Xception
import tensorflow.keras.backend as K

from src.models.mobilenet_v3_large import MobileNetV3_Large
from src.models.mobilenet_v3_small import MobileNetV3_Small

class Attention(Layer):
    def __init__(self, size, **kwargs):
        self.trainable=True
        self.size = size
        super(Attention, self).__init__(**kwargs)
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'size': self.size,
        })
        return config
    
    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q = self.add_weight(name='q',
                                shape=(1, self.size),
                                initializer='ones',
                                trainable=self.trainable)

    def call(self, x):
        stream1, stream2 = x[0], x[1]
        
        d1 = Lambda(lambda x: K.sum(x * self.q, axis=1, keepdims=True))(stream1) # sum over second axis
        d2 = Lambda(lambda x: K.sum(x * self.q, axis=1, keepdims=True))(stream2)
        ds = Concatenate(axis=1)([d1, d2])

        # d1 and d2 and of size (bs, 1) individually
        # ds of size (bs, 2)

        tmp = Softmax(axis=0)(ds)
        w1 = Lambda(lambda x: x[:, 0])(tmp)
        w2 = Lambda(lambda x: x[:, 1])(tmp)

        w1 = Lambda(lambda x: K.expand_dims(x, -1))(w1)
        w2 = Lambda(lambda x: K.expand_dims(x, -1))(w2)

        stream1 = Lambda(lambda x: x[0]*x[1])([stream1, w1])
        stream2 = Lambda(lambda x: x[0]*x[1])([stream2, w2])
        result = Add()([stream1, stream2])
        
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0]

def attention_model(num_classes, backbone = 'MobileNetV3_Small', shape=(256, 256, 3)):
    if backbone == 'Xception':
        stream1 = Xception(include_top=False, weights='imagenet', input_shape=shape)
        stream2 = Xception(include_top=False, weights='imagenet', input_shape=shape)
    elif backbone == 'MobileNetV3_Large':
        stream1 = MobileNetV3_Large(shape, num_classes).build()
        stream2 = MobileNetV3_Large(shape, num_classes).build()
    else: # MobileNetV3_Small
        stream1 = MobileNetV3_Small(shape, num_classes).build()
        stream2 = MobileNetV3_Small(shape, num_classes).build()

    input1 = Input(shape)
    input2 = Input(shape)
    output1 = stream1(input1)
    output2 = stream2(input2)
   
    # rename module so that model checkpoint can save it
    stream1._name = "stream1"
    for w in stream1.weights:
        w._handle_name = 'stream1_' + w.name
    stream2._name = "stream2"
    for w in stream2.weights:
        w._handle_name = 'stream2_' + w.name
        
    if backbone == 'Xception':
        output1 = GlobalAveragePooling2D(name='avg_pool_1')(output1)
        output2 = GlobalAveragePooling2D(name='avg_pool_2')(output2)

    output = Attention(size=output1.shape[1])([output1, output2])

    if num_classes==1:
        output = Dense(num_classes, activation='sigmoid', name='predictions')(output)
    else:
        output = Dense(num_classes, activation='softmax', name='predictions')(output)

    return Model(inputs=[input1, input2], outputs=output)

if __name__ == "__main__":
    model = attention_model(2)
    print(model.summary())
    from keras.utils import plot_model
    plot_model(model, 'model.png', show_shapes=True)

