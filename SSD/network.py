import tensorflow as tf

def ssd_model():
    vgg_model = tf.keras.applications.VGG16(include_top = False, input_shape = (300, 300, 3))
    vgg_model = tf.keras.Model(inputs = vgg_model.input, outputs = vgg_model.get_layer('block5_conv3').output)

    cov_out_1 = vgg_model.output
    x = tf.keras.layers.MaxPooling2D(padding = 'same')(cov_out_1)
    x = tf.keras.layers.Conv2D(1024, 3, padding = 'same', activation = 'relu')(x)
    cov_out_2 = tf.keras.layers.Conv2D(1024, 1, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.MaxPooling2D(padding = 'same')(cov_out_2)
    x = tf.keras.layers.Conv2D(256, 1, padding = 'same', activation = 'relu')(x)
    cov_out_3 = tf.keras.layers.Conv2D(512, 3, strides = 2, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.MaxPooling2D(padding = 'same')(cov_out_3)
    x = tf.keras.layers.Conv2D(128, 1, padding = 'same', activation = 'relu')(x)
    cov_out_4 = tf.keras.layers.Conv2D(256, 3, strides = 2, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.MaxPooling2D(padding = 'same')(cov_out_4)
    x = tf.keras.layers.Conv2D(128, 1, padding = 'same', activation = 'relu')(x)
    cov_out_5 = tf.keras.layers.Conv2D(256, 3, strides = 1, padding = 'same', activation = 'relu')(x)

    # 9 boxes * coordinates (4). Filter of 9 * 4 = depth.
    reg1 = tf.keras.layers.Conv2D(9 * 4, 3, padding = 'same')(cov_out_1)
    reg2 = tf.keras.layers.Conv2D(9 * 4, 3, padding = 'same')(cov_out_2)
    reg3 = tf.keras.layers.Conv2D(9 * 4, 3, padding = 'same')(cov_out_3)
    reg4 = tf.keras.layers.Conv2D(9 * 4, 3, padding = 'same')(cov_out_4)
    reg5 = tf.keras.layers.Conv2D(9 * 4, 3, padding = 'same')(cov_out_5)

    # Classification (9 boxes of 5 different classes (multiclass) and index = 0 for neg class)
    clf1 = tf.keras.layers.Conv2D(9 * 6, 3, padding = 'same')(cov_out_1)
    clf2 = tf.keras.layers.Conv2D(9 * 6, 3, padding = 'same')(cov_out_2)
    clf3 = tf.keras.layers.Conv2D(9 * 6, 3, padding = 'same')(cov_out_3)
    clf4 = tf.keras.layers.Conv2D(9 * 6, 3, padding = 'same')(cov_out_4)
    clf5 = tf.keras.layers.Conv2D(9 * 6, 3, padding = 'same')(cov_out_5)

    regr_list = [tf.keras.layers.Flatten()(i) for i in [reg1, reg2, reg3, reg4, reg5]]
    clf_list = [tf.keras.layers.Flatten()(i) for i in [clf1, clf2, clf3, clf4, clf5]]

    regression = tf.keras.layers.concatenate(regr_list)
    classification = tf.keras.layers.concatenate(clf_list)

    return tf.keras.Model(vgg_model.input, [regression, classification])