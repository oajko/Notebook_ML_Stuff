import tensorflow as tf
import numpy as np

def classification_loss(y_true, y_pred):
    true_mask = y_true > 0
    return tf.math.square(y_true[true_mask] - y_pred[true_mask])

# Input of nxnx125 where per 25 = new anchor box
def detection_loss(y_true, y_pred):
    true_mask = y_true[np.arange(4, 125, 25)] > 0
    true_mask = np.repeat(true_mask, 25)

    false_mask = y_true[np.arange(4, 125, 25)] == 0 & y_true[np.arange(3, 125, 25)] > 0
    false_mask = np.repeat(false_mask, 25)

    x_loss = tf.math.square(y_true[true_mask][np.arange(0, 125, 25)] - y_pred[true_mask][np.arange(0, 125, 25)])
    y_loss = tf.math.square(y_true[true_mask][np.arange(1, 125, 25)] - y_pred[true_mask][np.arange(1, 125, 25)])
    w_loss = tf.math.square(y_true[true_mask][np.arange(2, 125, 25)] - y_pred[true_mask][np.arange(2, 125, 25)])
    h_loss = tf.math.square(y_true[true_mask][np.arange(3, 125, 25)] - y_pred[true_mask][np.arange(3, 125, 25)])

    bb_loss = x_loss, y_loss, w_loss, h_loss

    pos_loss = tf.math.square(y_true[true_mask][np.arange(4, 125, 25)] - y_pred[true_mask][np.arange(4, 125, 25)])
    neg_loss = tf.math.square(y_true[false_mask][np.arange(4, 125, 25)] - y_pred[false_mask][np.arange(4, 125, 25)])

    obj_loss = pos_loss + neg_loss

    class_idxs = np.vstack([np.arange(i - 20, i, 1) for i in range(0, 125, 25)]).flatten()

    prob_loss = tf.math.square(y_true[true_mask][class_idxs] - y_pred[true_mask][class_idxs])

    return bb_loss + obj_loss + prob_loss