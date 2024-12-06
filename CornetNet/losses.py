import tensorflow as tf

# nxm map wherein values are embedding for each corner
def embedding_loss(tl_true, tl_pred, br_true, br_pred):
    tl = tf.where(tl_true > 0, tl_pred, 0)
    br = tf.where(br_true > 0, br_pred, 0)
    tl_size = tf.reduce_sum(tf.cast(tl > 0, tf.float32))
    br_size = tf.reduce_sum(tf.cast(br > 0, tf.float32))

    tl_ek = tf.reduce_sum(tl) / (tl_size + 1e-4)
    br_ek = tf.reduce_sum(br) / (br_size + 1e-4)
    ek = tl_ek + br_ek

    pull_tl = tf.reduce_sum(tf.square(tl - ek)) / (tl_size + 1e-4)
    pull_br = tf.reduce_sum(tf.square(br - ek)) / (br_size + 1e-4)
    pull = pull_tl + pull_br
    
    push = tf.reduce_sum(tf.maximum(0.0, 1 - tf.abs(tl_ek - br_ek)))
    loss = pull + push
    return loss

# etk = tl, ebk = br,
# pull to train, push to seperate corners. ek = avg ebk, etk

def heatmap_loss(y_true, y_pred):
    alpha = 0.1
    beta = 0.1

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    pos_loss = (1 - y_pred)**alpha * tf.math.log(y_pred)
    neg_loss = (1 - y_true)**beta * y_pred**alpha * tf.math.log(1 - y_pred)

    loss = tf.where(tf.equal(y_true, 1), pos_loss, neg_loss)
    return -tf.reduce_mean(loss)


def offset_loss(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    loss = tf.where(diff < 1, 0.5 * tf.square(diff), diff - 0.5)
    return tf.reduce_mean(loss)


def total_loss(y_true, y_pred):

    tl_heat, tl_embed, tl_off, br_heat, br_embed, br_off = y_true[..., 0: 9], y_true[..., 9], y_true[..., 10: 12], y_true[..., 12: 20], y_true[..., 20], y_true[..., 21: 23] 
    ptl_heat, ptl_embed, ptl_off, pbr_heat, pbr_embed, pbr_off = y_pred[..., 0: 9], y_pred[..., 9], y_pred[..., 10: 12], y_pred[..., 12: 20], y_pred[..., 20], y_pred[..., 21: 23]

    tl_heat_loss = heatmap_loss(tl_heat, ptl_heat)
    br_heat_loss = heatmap_loss(br_heat, pbr_heat)

    tl_off_loss = offset_loss(tl_off, ptl_off)
    br_off_loss = offset_loss(br_off, pbr_off)

    embed_loss = embedding_loss(tl_embed, ptl_embed, br_embed, pbr_embed)

    return tl_heat_loss + br_heat_loss + tl_off_loss + br_off_loss + embed_loss