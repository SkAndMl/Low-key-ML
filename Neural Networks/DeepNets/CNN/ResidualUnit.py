import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,GlobalAvgPool2D,MaxPooling2D,Flatten,Dense,Activation

class ResidualUnit(tf.keras.layers.Layer):

    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            Conv2D(filters, 3, strides=strides, padding="same"),
            BatchNormalization(),
            self.activation,
            Conv2D(filters, 3, strides=1, padding="same"),
            BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                Conv2D(filters, 1, strides=strides, padding="same"),
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
