def build_lenet5(input_shape,optimizer="nadam",
                      padding="same"):

    """
    LeNet5 CNN model espescially used for digit recognition
    :param input_shape: shape of the training instances: [height,width,channels]
    :param activation_conv: activation function for the Conv2D layers
    :param dropout_rate: specifies the fraction of neurons to be dropped
    :param activation_dense: activation function for the dense layers
    :param optimizer: optimizer for compiling the model
    :param padding: "same" or "valid"
    :param pool_size: specifies the pool_size for MaxPooling2D
    :return: returns a CNN model built based on the LeNet5 architecture adapted from Yann LeCun's paper
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Flatten, Conv2D, AveragePooling2D,Dense,Activation

    tf.random.set_seed(42)

    lenet5 = tf.keras.models.Sequential([
        Conv2D(filters=6, kernel_size=5, input_shape=input_shape, strides=1, activation="tanh"),
        AveragePooling2D(pool_size=2, strides=2),
        Activation("tanh"),
        Conv2D(filters=16, kernel_size=5, strides=1, activation="tanh", padding=padding),
        AveragePooling2D(pool_size=2, strides=2),
        Activation("tanh"),
        Conv2D(filters=120, kernel_size=5, strides=1, activation="tanh", padding=padding),
        Flatten(),
        Dense(84, activation="tanh"),
        Dense(10, activation="sigmoid")
    ], name="lenet5")

    lenet5.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics="accuracy")

    return lenet5

