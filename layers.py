import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization

@tf.keras.utils.register_keras_serializable()
class ConvRepeat(tf.keras.layers.Layer):
    def __init__(self, depth, kernel, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.kernel = kernel
        self.layers_list = tf.keras.Sequential()

        for _ in range(2):
            self.layers_list.add(Conv2D(depth, kernel, activation = 'relu'))
            self.layers_list.add(BatchNormalization())

    def call(self, inputs):
        return self.layers_list(inputs)
    
    def get_config(self):
        config = super(ConvRepeat, self).get_config()
        config.update
        (
            {
                'units': self.depth,
                'kernel': self.kernel 
            }
        )

@tf.keras.utils.register_keras_serializable()
class ConvSingle(tf.keras.layers.Layer):
    def __init__(self, depth, kernel, padding, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.kernel = kernel
        self.padding = padding
        self.conv = Conv2D(depth, kernel, padding = padding, activation = 'relu')

    def call(self, inputs):
        return self.conv(inputs)
    
    def get_config(self):
        config = super(ConvSingle, self).get_config()
        config.update
        (
            {
                'units': self.depth,
                'padding': self.padding,
                'kernel': self.kernel 
            }
        )