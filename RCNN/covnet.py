import tensorflow as tf

def cnn_features(layer_loop = 0, train = False, path = None):
    if train is True:
        input_layer = tf.keras.applications.MobileNet(input_shape = (224, 224, 3), include_top = False)
        input_layer.trainable = False
        for i in range(0, -layer_loop, -1):
            input_layer.layers[i].trainable = True
        x = input_layer.output
        x = tf.keras.layers.Flatten(name = 'flat')(x)
        x = tf.keras.layers.Dense(512, activation = 'relu')(x)
        output = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(input_layer.input, output)
    else:
        model = tf.keras.models.load_model(path)
        model_layers = model.layers
        for i in range(-1, -len(model_layers), -1):
            if model_layers[i].name == 'flat':
                break
        model = tf.keras.Model(inputs = model.input, outputs = model.layers[i].output)
    return model

# train covnet -> Flattened covnet preds -> train SVM -> 
