import warnings
warnings.filterwarnings("ignore")

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, \
    Add, Lambda, Layer, Concatenate, Softmax
from keras.applications.xception import Xception
import keras.backend as K

from src.models.mobilenet_v3_large import MobileNetV3_Large
from src.models.mobilenet_v3_small import MobileNetV3_Small

class Attention(Layer):
    def __init__(self, size, **kwargs):
        self.trainable=True
        self.size = size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q = self.add_weight(name='q',
                                shape=(1, self.size),
                                initializer='ones',
                                trainable=self.trainable)

    def call(self, x):
        stream1, stream2 = x[0], x[1]

        d1 = K.sum(stream1 * self.q, axis=1, keepdims=True) # sum over second axis
        d2 = K.sum(stream2 * self.q, axis=1, keepdims=True)
        print(d1.shape, d2.shape)
        ds = Concatenate(axis=1)([d1, d2])

        # d1 and d2 and of size (bs, 1) individually
        # ds of size (bs, 2)

        tmp = Softmax(axis=0)(ds)
        w1 = tmp[:, 0]
        w2 = tmp[:, 1]

        w1 = Lambda(lambda x: K.expand_dims(x, -1))(w1)
        w2 = Lambda(lambda x: K.expand_dims(x, -1))(w2)

        stream1 = Lambda(lambda x: x[0]*x[1])([stream1, w1])
        stream2 = Lambda(lambda x: x[0]*x[1])([stream2, w2])
        result = Add()([stream1, stream2])
        
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0]

def attention_model(classes, backbone = 'MobileNetV3_Small', shape=(256, 256, 3)):
    if backbone == 'Xception':
        stream1 = Xception(include_top=False, weights='imagenet', input_shape=shape)
        stream2 = Xception(include_top=False, weights='imagenet', input_shape=shape)
    elif backbone == 'MobileNetV3_Large':
        stream1 = MobileNetV3_Large(shape, classes).build()
        stream2 = MobileNetV3_Large(shape, classes).build()
    else: # MobileNetV3_Small
        stream1 = MobileNetV3_Small(shape, classes).build()
        stream2 = MobileNetV3_Small(shape, classes).build()

    input1 = Input(shape)
    input2 = Input(shape)
    output1 = stream1(input1)
    output2 = stream2(input2)
   
    stream1._name = "stream1"
    stream2._name = "stream2"
    if backbone == 'Xception':
        output1 = GlobalAveragePooling2D(name='avg_pool_1')(output1)
        output2 = GlobalAveragePooling2D(name='avg_pool_2')(output2)

    output = Attention(size=output1.shape[1])([output1, output2])

    if classes==1:
        output = Dense(classes, activation='sigmoid', name='predictions')(output)
    else:
        output = Dense(classes, activation='softmax', name='predictions')(output)

    return Model(inputs=[input1, input2], outputs=output)

if __name__ == "__main__":
    model = attention_model(2)
    print(model.summary())
    from keras.utils import plot_model
    plot_model(model, 'model.png', show_shapes=True)

