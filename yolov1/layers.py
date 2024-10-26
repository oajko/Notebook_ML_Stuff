import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

@tf.keras.utils.register_keras_serializable()
class SingleConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, strides, pool = False, **kwargs):
        super(SingleConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.pool = pool

        self.conv = Conv2D(filters, (kernel, kernel), (strides, strides), activation = 'leaky_relu', padding = 'same')
        self.batch = BatchNormalization()
        self.pooling = MaxPooling2D(strides = 2) if pool is True else None
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch(x)
        if self.pool is True:
            x = self.pooling(x)
        return x
    
    def get_config(self):
        config = super(SingleConv, self).get_config()
        config.update({
                'filters': self.filters,
                'kernel': self.kernel,
                'strides': self.strides,
                'pool': self.pool
            })
        return config

@tf.keras.utils.register_keras_serializable()
class RepeatConv(tf.keras.layers.Layer):
    def __init__(self, filters, loops, **kwargs):
        super(RepeatConv, self).__init__(**kwargs)
        self.filters = filters
        self.loop = loops
        self.layers = tf.keras.Sequential()

        for _ in range(loops):
            self.layers.add(Conv2D(filters // 2, (1, 1), padding = 'same', activation = 'leaky_relu'))
            self.layers.add(BatchNormalization())
            self.layers.add(Conv2D(filters, (3, 3), padding = 'same', activation = 'leaky_relu'))
            self.layers.add(BatchNormalization())
    
    def call(self, inputs):
        return self.layers(inputs)
    
    def get_config(self):
        config = super(RepeatConv, self).get_config()
        config.update({
                'filters': self.filters,
                'loops': self.loop
            })
        return config
