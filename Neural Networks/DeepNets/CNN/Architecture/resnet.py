import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,GlobalAvgPool2D,MaxPooling2D,Flatten,Dense,Activation

class ResidualUnit(tf.keras.layers.Layer):

    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            Conv2D(filters, 3, strides=strides, padding="same",use_bias=False),
            BatchNormalization(),
            self.activation,
            Conv2D(filters, 3, strides=1, padding="same",use_bias=False),
            BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                Conv2D(filters, 1, strides=strides, padding="same",use_bias=False),
                BatchNormalization()
            ]

    def call(self, X):
        Z = X
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = X
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

def build_resnet(input_shape,optimizer):
    resnet = tf.keras.models.Sequential()
    resnet.add(Conv2D(64,7,2,input_shape=input_shape,padding="same"))
    resnet.add(BatchNormalization())
    resnet.add(Activation("relu"))
    resnet.add(MaxPooling2D(pool_size=3,strides=2,padding="same"))
    prev_filters = 64
    for filters in [64]*3 + [128]*4 + [256]*6 + [512]*3:
        strides = 1 if filters==prev_filters else 2
        resnet.add(ResidualUnit(filters,strides=strides))
        prev_filters = filters
    resnet.add(GlobalAvgPool2D())
    resnet.add(Flatten())
    resnet.add(Dense(10,activation="softmax"))

    resnet.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,
               metrics="accuracy")

    return resnet
