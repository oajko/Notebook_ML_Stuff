import tensorflow as tf
import loss_funcs


# input is tensor of n,n, (bb * (5) + class_num)

# Detect pos and no class gt -> then simple logic
# detect two things -> bounding box in same pos and the pos itself.

# t, x, y, w, h, c


# identify 1 class
def yolo_loss(y_true, y_pred):
    one_mask = tf.equal(y_true[..., 0], 1)
    zero_mask = tf.equal(y_true[..., 0], 0)

    coord_loss = 5 * tf.math.reduce_sum(tf.math.square(y_true[one_mask][..., 1: 3] - y_pred[one_mask][..., 1: 3]))
    dim_loss = 5 * tf.math.reduce_sum(tf.math.square(y_true[one_mask][..., 3: 5] - y_pred[one_mask][..., 3: 5]))
    localization_loss = coord_loss + dim_loss

    iou = loss_funcs.iou(y_true[one_mask], y_pred[one_mask])

    conf_loss = tf.math.reduce_sum(tf.math.square(iou - iou * y_pred[one_mask][..., 5]))
    noob_loss = 0.5 * tf.math.reduce_sum(tf.math.square(y_pred[zero_mask][..., 5]))

    class_loss = tf.math.reduce_sum(tf.math.square(y_true[one_mask][..., 5] - y_pred[one_mask][..., 5]))
    
    return localization_loss + conf_loss + noob_loss + class_loss

