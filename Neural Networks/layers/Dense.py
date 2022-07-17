import tensorflow as tf

class Dense(tf.keras.layers.Layer):

    def __init__(self,units,activation,**kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    def build(self,batch_input_shape):
        self.kernel = self.add_weight(name="kernel",
                                    shape=[batch_input_shape[-1],self.units],
                                    initializer="glorot_normal")
        self.bias = self.add_weight(name="bias",
                                    shape=[self.units],
                                    initializer="zeros")
        super().build(batch_input_shape)
    
    def call(self,X):
        return self.activation(X@self.kernel + self.bias)
    
    def compute_output_shape(self,batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1]+[self.units])
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config,"units":self.units,
                "activation" : tf.keras.activations.serialize(self.activation)}
