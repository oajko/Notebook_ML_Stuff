import tensorflow as tf

def hour_glass(input_tensor):
    x = residual_block(input_tensor, 256, 384)
    x = residual_block(x, 384, 384)
    x = residual_block(x, 384, 512)
    x = upsample_block(x, 384)
    x = upsample_block(x, 384)
    x = upsample_block(x, 256)
    return x

# Optinal relu after BN
def residual_block(input_tensor, channel_size, pool_size):
    x = tf.keras.layers.Conv2D(channel_size, 3, strides = 1, padding = 'same', activation = 'relu')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(channel_size, 3, strides = 1, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Pooling
    x = tf.keras.layers.Conv2D(pool_size, 3, strides = 2, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def upsample_block(input_tensor, channel_size):
    x = tf.keras.layers.UpSampling2D(size = (2, 2))(input_tensor)
    x = tf.keras.layers.Conv2D(channel_size, 3, strides = 1, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(channel_size, 3, strides = 1, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

class HorizontalPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(HorizontalPooling, self).__init__(**kwargs)

    def call(self, inputs):
        x = tf.transpose(tf.scan(lambda prev, curr: tf.maximum(prev, curr), tf.transpose(inputs, [0, 1, 3, 2])), [0, 1, 3, 2])
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
class VerticalPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(VerticalPooling, self).__init__(**kwargs)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        x = tf.transpose(tf.scan(lambda prev, curr: tf.maximum(prev, curr), tf.transpose(inputs, [0, 2, 3, 1])), [0, 3, 1, 2])
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape

def corner_block(input_tensor):
    horizontal = tf.keras.layers.Conv2D(256, 3, padding = 'same', activation = 'relu')(input_tensor)
    horizontal = tf.keras.layers.BatchNormalization()(horizontal)
    horizontal = HorizontalPooling()(horizontal)

    vertical = tf.keras.layers.Conv2D(256, 3, padding = 'same', activation = 'relu')(input_tensor)
    vertical = tf.keras.layers.BatchNormalization()(vertical)
    vertical = VerticalPooling()(vertical)

    tl_comb = horizontal + vertical
    tl_output = tf.keras.layers.Conv2D(256, 3, padding = 'same', activation = 'relu')(tl_comb)

    pass_through = tf.keras.layers.Conv2D(256, 1, padding = 'same', activation = 'relu')(input_tensor)
    pass_through = tf.keras.layers.BatchNormalization()(pass_through)

    x = pass_through + tl_output
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(256, 3, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Coordinates. Suppose each img is nxmxc
    heatmap = tf.keras.layers.Conv2D(256, 3, padding = 'same', activation = 'relu')(x)
    heatmap = tf.keras.layers.Conv2D(8, 1, padding = 'same', activation = 'softmax')(heatmap)

    # Distance between embeddings - 1D for each detected corner. Distance measure wherein small for same object. nxmx1
    embedding = tf.keras.layers.Conv2D(256, 3, padding = 'same', activation = 'relu')(x)
    embedding = tf.keras.layers.Conv2D(1, 1, padding = 'same', activation = 'sigmoid')(embedding)
    
    offset = tf.keras.layers.Conv2D(256, 3, padding = 'same', activation = 'relu')(x)
    offset = tf.keras.layers.Conv2D(2, 1, padding = 'same', activation = 'softmax')(offset)

    return heatmap, embedding, offset

def network():
    input_layer = tf.keras.layers.Input((512, 512, 3))
    init_down = tf.keras.layers.Conv2D(128, 7, strides = 2, padding = 'same', activation = 'relu')(input_layer)
    input_data = tf.keras.layers.Conv2D(256, 3, strides = 2, padding = 'same', activation = 'relu')(init_down)

    # Hourglass + outputs
    x = hour_glass(input_data)
    hour_out = tf.keras.layers.Conv2D(256, 1, padding = 'same', activation = 'relu')(x)
    hour_out = tf.keras.layers.BatchNormalization()(hour_out)

    # Skip connection
    input_skip = tf.keras.layers.Conv2D(256, 1, padding = 'same', activation = 'relu')(input_data)
    input_skip = tf.keras.layers.BatchNormalization()(input_skip)

    hour_out = hour_out + input_skip

    # No mention of skip connection in second hourglass
    x = tf.keras.layers.Conv2D(256, 3, padding = 'same', activation = 'relu')(hour_out)
    x = hour_glass(hour_out)

    tl_heatmap, tl_embedding, tl_offset = corner_block(x)
    br_heatmap, br_embedding, br_offset = corner_block(x)

    # Concatenate for loss func
    output_ = tf.keras.layers.concatenate([tl_heatmap, tl_embedding, tl_offset, br_heatmap, br_embedding, br_offset], axis = -1)

    return tf.keras.Model(input_layer, output_)
