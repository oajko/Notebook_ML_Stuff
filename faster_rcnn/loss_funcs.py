import tensorflow as tf

def rpn_loss(y_true, y_pred):
    reg_true, class_true = y_true
    reg_pred, class_pred = y_pred

    class_loss = tf.keras.losses.BinaryCrossEntropy()(class_true, class_pred)

    pos_mask = tf.where(class_true[..., 1] > 0)

    reg_true_pos = tf.gather_nd(reg_true, pos_mask)
    reg_pred_pos = tf.gather_nd(reg_pred, pos_mask)

    regr_loss = tf.keras.losses.Huber()(reg_true_pos, reg_pred_pos)

    return regr_loss + class_loss