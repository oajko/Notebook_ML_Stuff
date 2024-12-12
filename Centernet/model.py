import tensorflow as tf

def hour_glass(inputs):
    x = downsample(inputs, 128, 256)
    x = downsample(x, 256, 256)
    x = downsample(x, 256, 384)

    x = upsample(x, 256)
    x = upsample(x, 256)
    x = upsample(x, 128)
    return x

def upsample(inputs, chanel_size):
    x = tf.keras.layers.UpSampling2D(size = (2, 2))(inputs)
    x = tf.keras.layers.Conv2D(chanel_size, 3, strides = 1, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def downsample(inputs, chanel_size, pool_size):
    x = tf.keras.layers.Conv2D(chanel_size, 3, strides = 1, padding = 'same', activation = 'relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(pool_size, 3, strides = 2, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

class LeftPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LeftPooling, self).__init__(**kwargs)
    def call(self, inputs):
        return tf.transpose(tf.scan(lambda a, x: tf.maximum(a, x), tf.transpose(inputs, [1, 0, 2, 3]), reverse = True), [1, 0, 2, 3])
    def compute_output_shape(self, input_shape):
        return input_shape

class RightPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RightPooling, self).__init__(**kwargs)
    def call(self, inputs):
        return tf.transpose(tf.scan(lambda a, x: tf.maximum(a, x), tf.transpose(inputs, [1, 0, 2, 3])), [1, 0, 2, 3])
    def compute_output_shape(self, input_shape):
        return input_shape

class UpPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(UpPooling, self).__init__(**kwargs)
    def call(self, inputs):
        return tf.transpose(tf.scan(lambda a, x: tf.maximum(a, x), tf.transpose(inputs, [2, 1, 0, 3]), reverse = True), [2, 1, 0, 3])
    def compute_output_shape(self, input_shape):
        return input_shape

class DownPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DownPooling, self).__init__(**kwargs)
    def call(self, inputs):
        return tf.transpose(tf.scan(lambda a, x: tf.maximum(a, x), tf.transpose(inputs, [2, 1, 0, 3])), [2, 1, 0, 3])
    def compute_output_shape(self, input_shape):
        return input_shape

# Diagram and description differ. I went with worded option
def center_pooling(inputs):
    center_pool_hori = tf.keras.layers.Conv2D(128, 3, padding = 'same', activation = 'relu')(inputs)
    center_pool_hori = tf.keras.layers.BatchNormalization()(center_pool_hori)
    left_pool = LeftPooling()(center_pool_hori)
    right_pool = RightPooling()(center_pool_hori)
    hori_max = tf.keras.layers.maximum([left_pool, right_pool])

    center_pool_vert = tf.keras.layers.Conv2D(128, 3, padding = 'same', activation = 'relu')(inputs)
    center_pool_vert = tf.keras.layers.BatchNormalization()(center_pool_vert)
    up_pool = UpPooling()(center_pool_vert)
    down_pool = DownPooling()(center_pool_vert)
    vert_max = tf.keras.layers.maximum([up_pool, down_pool])
    return hori_max + vert_max

def left_cascade_pooling(inputs):
    left = tf.keras.layers.Conv2D(128, 3, padding = 'same', activation = 'relu')(inputs)
    right = tf.keras.layers.Conv2D(128, 3, padding = 'same', activation = 'relu')(inputs)
    left = LeftPooling()(left)
    left_pooled = left + right
    x = tf.keras.layers.Conv2D(128, 3, padding = 'same', activation = 'linear')(left_pooled)
    x = tf.keras.layers.BatchNormalization()(x)
    up_pooled = UpPooling()(x)
    return left_pooled + up_pooled

def right_cascade_pooling(inputs):
    left = tf.keras.layers.Conv2D(128, 3, padding = 'same', activation = 'relu')(inputs)
    right = tf.keras.layers.Conv2D(128, 3, padding = 'same', activation = 'relu')(inputs)
    left = RightPooling()(left)
    right_pooled = left + right
    x = tf.keras.layers.Conv2D(128, 3, padding = 'same', activation = 'linear')(right_pooled)
    x = tf.keras.layers.BatchNormalization()(x)
    down_pooled = DownPooling()(x)
    return right_pooled + down_pooled

def output_layers(inputs):
    heatmap = tf.keras.layers.Conv2D(128, 3, padding = 'same', activation = 'relu')(inputs)
    heatmap = tf.keras.layers.Conv2D(8, 3, padding = 'same', activation = 'softmax')(heatmap)
    offset = tf.keras.layers.Conv2D(128, 3, padding = 'same', activation = 'relu')(inputs)
    offset = tf.keras.layers.Conv2D(2, 3, padding = 'same', activation = 'linear')(offset)
    embedding = tf.keras.layers.Conv2D(128, 3, padding = 'same', activation = 'relu')(inputs)
    embedding = tf.keras.layers.Conv2D(1, 3, padding = 'same', activation = 'sigmoid')(embedding)
    return heatmap, offset, embedding

def model():
    input_layer = tf.keras.layers.Input((256, 256, 3))
    x = tf.keras.layers.Conv2D(64, 7, strides = 2, padding = 'same', activation = 'relu')(input_layer)
    x = tf.keras.layers.Conv2D(128, 3, strides = 2, padding = 'same', activation = 'relu')(x)
    x = hour_glass(x) + x
    tl_pool = left_cascade_pooling(x)
    br_pool = right_cascade_pooling(x)
    center = center_pooling(x)

    tl_heat, tl_off, tl_embed = output_layers(tl_pool)
    br_heat, br_off, br_embed = output_layers(br_pool)
    cen_heat, cen_off, _ = output_layers(center)

    output_layer = tf.keras.layers.concatenate([tl_heat, tl_off, tl_embed, br_heat, br_off, br_embed, cen_heat, cen_off])

    return tf.keras.Model(input_layer, output_layer)