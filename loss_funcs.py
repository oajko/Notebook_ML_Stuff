import tensorflow as tf


# input of t, x, y, w, h
# Suppose filtering is done and it's only zero or one class tensor

# input is boolean mask, so tensor of [t, x, y, w, h]. i.e., [[t, x, y, w, h], [t, x, y, w, h]]
# tf reduce sum

def iou(b1, b2):
    # tensor of pos
    left_x1 = tf.math.subtract(b1[..., 1], b1[..., 3] / 2)
    right_x1 = tf.math.add(b1[..., 1], b1[..., 3])
    top_x1 = tf.math.add(b1[..., 2], b1[..., 4] / 2)
    bot_x1 = tf.math.subtract(b1[..., 2], b1[..., 4] / 2)

    left_x2 = tf.math.subtract(b2[..., 1], b2[..., 3] / 2)
    right_x2 = tf.math.add(b2[..., 1], b2[..., 3])
    top_x2 = tf.math.add(b2[..., 2], b2[..., 4] / 2)
    bot_x2 = tf.math.subtract(b2[..., 2], b2[..., 4] / 2)

    width = tf.math.subtract(tf.math.minimum(right_x1, right_x2), tf.math.maximum(left_x1, left_x2))
    height = tf.math.subtract(tf.math.minimum(top_x1, top_x2), tf.math.maximum(bot_x1, bot_x2))

    intersection = width * height

    sq_1 = tf.math.subtract(right_x1, left_x1) * tf.math.subtract(top_x1, bot_x1)
    sq_2 = tf.math.subtract(right_x2, left_x2) * tf.math.subtract(top_x2, bot_x2)

    union = (sq_1 + sq_2 - intersection) + 1e-6

    # return tf.math.reduce_sum(tf.math.divide(intersection, union))
    return tf.math.divide(intersection, union)

