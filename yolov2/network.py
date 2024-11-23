import tensorflow as tf

# Classification 224x224 and finetune with 448x448

# Using relu after batch norm. Not mentioned before ot after, but I don't think it matters much
def classification_train(input_size, output_size, path = None):
    # Used to redo training (i.e., paused with last saved weights). Only accepting model, not weights.
    if path is not None:
        return tf.keras.models.load_model(path)
    
    input_layer = tf.keras.layers.Input(input_size)
    x = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x)
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x)

    x = tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = 1, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x)

    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters = 128, kernel_size = 1, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x)

    x = tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = 1, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = 1, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(name = 'collapse_covnet')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x)

    x = tf.keras.layers.Conv2D(filters = 1024, kernel_size = 3, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters = 512, kernel_size = 1, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters = 1024, kernel_size = 3, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters = 512, kernel_size = 1, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters = 1024, kernel_size = 3, strides = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters = 1000, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', name = 'final_covnet_layer')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output_layer = tf.keras.layers.Dense(output_size, activation = 'softmax')

    return tf.keras.Model(input_layer, output_layer)

def classification_finetune(path):
    model = tf.keras.models.load_models(path, compile = False)
    model.layers.trainable = False
    network = model.layers[1:]

    input_layer = tf.keras.layers.Input((448, 448, 3))
    new_model = network(input_layer)

    # Trains final 8 layers (after batch + relu)
    for layer in range(-24, 0, 1):
        new_model.layers[layer].trainable = True
    return new_model

def joint_model(path):
    temp_model = tf.keras.models.load_model(path, compile = False)
    c = 0
    for idx, layer in enumerate(temp_model.layers):
        if layer.name == 'final_covnet_layer':
            break
        if layer.name == 'collapse_covnet':
            c = idx
    # Make dynamic input size
    input_layer = tf.keras.layers.Input((None, None, 3))
    x = input_layer
    for layer in temp_model.layers[1: idx + 1]:
        x = layer(x)

    concat_layer = temp_model.layers[c].output
    concat_layer = tf.reshape(concat_layer, (-1, 13, 13, 2048))

    concat = tf.keras.layers.Concatenate(axis = -1)([x, concat_layer])
    concat_2 = tf.keras.Conv2D(1024, 3, padding = 'same', activation = 'relu')(concat)
    # 5 k means (boxes), 5 coords and 20 classes.
    linear = tf.keras.conv2D(5 * 5 + 5 * 20, 1, padding = 'same', activation = 'linear')(concat_2)

    # Classification (original model above func)
    classification = temp_model
    
    return tf.keras.Model(input_layer, [linear, classification])

