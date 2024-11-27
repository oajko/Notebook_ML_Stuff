import tensorflow as tf

def class_loss(y_true, y_pred):
    dim_37 = 37 * 37 * 6
    dim_18 = 18 * 18 + dim_37
    dim_9 = 9 * 9 + dim_18
    dim_5 = 5 * 5 + dim_9
    dim_3 = 3 * 3 + dim_5
    dim_2 = 2 * 2 + dim_3

    # Reshape flattened gt back to normal. I believe tf loss runs per sample, not batch
    t_box_37 = y_true[:dim_37].reshape((37, 37, 6 * 9))
    t_box_18 = y_true[dim_37: dim_18].reshape((18, 18, 6 * 9))
    t_box_9 = y_true[dim_18: dim_9].reshape((9, 9, 6 * 9))
    t_box_5 = y_true[dim_9: dim_5].reshape((5, 5, 6 * 9))
    t_box_3 = y_true[dim_5: dim_3].reshape((3, 3, 6 * 9))
    t_box_2 = y_true[dim_3: dim_2].reshape((2, 2, 6 * 9))
    t_box_1 = y_true[dim_2: dim_2 + 1].reshape((1, 1, 6 * 9))

    p_box_37 = y_pred[:dim_37].reshape((37, 37, 6 * 9))
    p_box_18 = y_pred[dim_37: dim_18].reshape((18, 18, 6 * 9))
    p_box_9 = y_pred[dim_18: dim_9].reshape((9, 9, 6 * 9))
    p_box_5 = y_pred[dim_9: dim_5].reshape((5, 5, 6 * 9))
    p_box_3 = y_pred[dim_5: dim_3].reshape((3, 3, 6 * 9))
    p_box_2 = y_pred[dim_3: dim_2].reshape((2, 2, 6 * 9))
    p_box_1 = y_pred[dim_2: dim_2 + 1].reshape((1, 1, 6 * 9))

# Could seperate to allow const for pos or neg class but it's fine.
    box37_pmask = (t_box_37[..., 0] == 1)
    box18_pmask = (t_box_18[..., 0] == 1)
    box9_pmask = (t_box_9[..., 0] == 1)
    box5_pmask = (t_box_5[..., 0] == 1)
    box3_pmask = (t_box_3[..., 0] == 1)
    box2_pmask = (t_box_2[..., 0] == 1)
    box1_pmask = (t_box_1[..., 0] == 1)

    negc37 = tf.math.log(tf.exp(p_box_37[box37_pmask]) / tf.sum(tf.exp(p_box_37[box37_pmask])))
    negc18 = tf.math.log(tf.exp(p_box_18[box18_pmask]) / tf.sum(tf.exp(p_box_18[box18_pmask])))
    negc9 = tf.math.log(tf.exp(p_box_9[box9_pmask]) / tf.sum(tf.exp(p_box_9[box9_pmask])))
    negc5 = tf.math.log(tf.exp(p_box_5[box5_pmask]) / tf.sum(tf.exp(p_box_5[box5_pmask])))
    negc3 = tf.math.log(tf.exp(p_box_3[box3_pmask]) / tf.sum(tf.exp(p_box_3[box3_pmask])))
    negc2 = tf.math.log(tf.exp(p_box_2[box2_pmask]) / tf.sum(tf.exp(p_box_2[box2_pmask])))
    negc1 = tf.math.log(tf.exp(p_box_1[box1_pmask]) / tf.sum(tf.exp(p_box_1[box1_pmask])))

    class_loss = negc37 + negc18 + negc9 + negc5 + negc3 + negc2 + negc1

    return class_loss

def regression_loss(y_true, y_pred):
    dim_37 = 37 * 37 * 6
    dim_18 = 18 * 18 + dim_37
    dim_9 = 9 * 9 + dim_18
    dim_5 = 5 * 5 + dim_9
    dim_3 = 3 * 3 + dim_5
    dim_2 = 2 * 2 + dim_3

    # Reshape flattened gt back to normal. I believe tf loss runs per sample, not batch
    t_box_37 = y_true[:dim_37].reshape((37, 37, 9 * 4))
    t_box_18 = y_true[dim_37: dim_18].reshape((18, 18, 9 * 4))
    t_box_9 = y_true[dim_18: dim_9].reshape((9, 9, 9 * 4))
    t_box_5 = y_true[dim_9: dim_5].reshape((5, 5, 9 * 4))
    t_box_3 = y_true[dim_5: dim_3].reshape((3, 3, 9 * 4))
    t_box_2 = y_true[dim_3: dim_2].reshape((2, 2, 9 * 4))
    t_box_1 = y_true[dim_2: dim_2 + 1].reshape((1, 1, 9 * 4))

    p_box_37 = y_pred[:dim_37].reshape((37, 37, 9 * 4))
    p_box_18 = y_pred[dim_37: dim_18].reshape((18, 18, 9 * 4))
    p_box_9 = y_pred[dim_18: dim_9].reshape((9, 9, 9 * 4))
    p_box_5 = y_pred[dim_9: dim_5].reshape((5, 5, 9 * 4))
    p_box_3 = y_pred[dim_5: dim_3].reshape((3, 3, 9 * 4))
    p_box_2 = y_pred[dim_3: dim_2].reshape((2, 2, 9 * 4))
    p_box_1 = y_pred[dim_2: dim_2 + 1].reshape((1, 1, 9 * 4))

    box37_pmask = (t_box_37 > 0)
    box18_pmask = (t_box_18 > 0)
    box9_pmask = (t_box_9 > 0)
    box5_pmask = (t_box_5 > 0)
    box3_pmask = (t_box_3 > 0)
    box2_pmask = (t_box_2 > 0)
    box1_pmask = (t_box_1 > 0)

    b37_x = tf.abs(tf.boolean_mask(p_box_37, box37_pmask) - tf.boolean_mask(t_box_37, box37_pmask))
    b37_loss = tf.where(b37_x < 1, 0.5 * tf.square(b37_x), 1 * (b37_x - 0.5 * 1))

    b18_x = tf.abs(tf.boolean_mask(p_box_18, box18_pmask) - tf.boolean_mask(t_box_18, box18_pmask))
    b18_loss = tf.where(b18_x < 1, 0.5 * tf.square(b18_x), 1 * (b18_x - 0.5 * 1))

    b9_x = tf.abs(tf.boolean_mask(p_box_9, box9_pmask) - tf.boolean_mask(t_box_9, box9_pmask))
    b9_loss = tf.where(b9_x < 1, 0.5 * tf.square(b9_x), 1 * (b9_x - 0.5 * 1))

    b5_x = tf.abs(tf.boolean_mask(p_box_5, box5_pmask) - tf.boolean_mask(t_box_5, box5_pmask))
    b5_loss = tf.where(b5_x < 1, 0.5 * tf.square(b5_x), 1 * (b5_x - 0.5 * 1))

    b3_x = tf.abs(tf.boolean_mask(p_box_3, box3_pmask) - tf.boolean_mask(t_box_3, box3_pmask))
    b3_loss = tf.where(b3_x < 1, 0.5 * tf.square(b3_x), 1 * (b3_x - 0.5 * 1))

    b2_x = tf.abs(tf.boolean_mask(p_box_2, box2_pmask) - tf.boolean_mask(t_box_2, box2_pmask))
    b2_loss = tf.where(b2_x < 1, 0.5 * tf.square(b2_x), 1 * (b2_x - 0.5 * 1))

    b1_x = tf.abs(tf.boolean_mask(p_box_1, box1_pmask) - tf.boolean_mask(t_box_1, box1_pmask))
    b1_loss = tf.where(b1_x < 1, 0.5 * tf.square(b1_x), 1 * (b1_x - 0.5 * 1))
    
    hub_loss = b37_loss + b18_loss + b9_loss + b5_loss + b3_loss + b2_loss + b1_loss

    return hub_loss