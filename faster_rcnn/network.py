import tensorflow as tf

# Model to fine_tune top_layer
def fine_tune_tl():
    mdl = tf.keras.applications.VGG16(input_shape = (224, 224, 3), include_top = False)
    input_layer = mdl.input
    mdl.trainable = False

    x = tf.keras.layers.Flatten(name = 'flatten_layer')(mdl.output)
    x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
    output = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)

    return tf.keras.Model(input_layer, output)

def fine_tune_cov(path):
    model = tf.keras.load_model(path)
    # Finetune last 10 layers
    for layer in range(0, -10, -1):
        model.layers[layer].trainable = True
    return model

# 128x128, 256x256, 512x512. 1:1, 1:2, 2:1. And scale down by 16 (div by 16)

def rpn(path):
    model = tf.keras.load_model(path)
    # Break out of pool layer
    fin = model.get_layer('block5_conv3').output
    model = tf.keras.Model(inputs = model.input, outputs = fin)
    model.trainable = False

    x = model.output
    x = tf.keras.layers.Conv2D(256, kernel_size = 3, padding = 'same', activation = 'relu')(x)

    num_anchors = 9
    regression  = tf.keras.layers.Conv2D(num_anchors * 4, kernel_size = 1, activation = 'linear')(x)
    classification  = tf.keras.layers.Conv2D(num_anchors * 2, kernel_size = 1, activation = 'sigmoid')(x)

    return tf.keras.Model(model.input, [regression, classification])