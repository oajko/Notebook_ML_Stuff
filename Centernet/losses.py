import tensorflow as tf

def total_loss(y_true, y_pred):
    ttl_heat, ttl_embed, ttl_off, tbr_heat, tbr_embed, tbr_off, tcen_heat, tcen_off = y_true[..., 0: 8], y_true[..., 8], y_true[..., 9: 11], y_true[..., 11: 19], y_true[..., 19],\
          y_true[..., 20: 22], y_true[..., 22: 30], y_true[..., 30: 32]
    ptl_heat, ptl_embed, ptl_off, pbr_heat, pbr_embed, pbr_off, pcen_heat, pcen_off = y_pred[..., 0: 8], y_pred[..., 8], y_pred[..., 9: 11], y_pred[..., 11: 19], y_pred[..., 19],\
          y_pred[..., 20: 22], y_pred[..., 22: 30], y_pred[..., 30: 32]
    hl_heat = heat_loss(ttl_heat, ptl_heat)
    hl_off = off_loss(ttl_off, ptl_off)
    br_heat = heat_loss(tbr_heat, pbr_heat)
    br_off = off_loss(tbr_off, pbr_off)
    embed = embed_loss(ttl_embed, ptl_embed, tbr_embed, pbr_embed)
    cen_heat = heat_loss(tcen_heat, pcen_heat)
    cen_off = off_loss(tcen_off, pcen_off)
    return hl_heat + hl_off + br_heat + br_off + embed + cen_heat + cen_off
    
def heat_loss(y_true, y_pred):
    alpha = 0.1
    beta = 0.1
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    pos_loss = (1 - y_pred)**alpha * tf.math.log(y_pred)
    neg_loss = (1 - y_true)**beta * y_pred**alpha * tf.math.log(1 - y_pred)
    loss = tf.where(tf.equal(y_true, 1), pos_loss, neg_loss)
    return -tf.reduce_mean(loss)

def embed_loss(l_true, l_pred, r_true, r_pred):
    tl = l_pred * l_true
    br = r_pred * r_true
    tl_size = tf.reduce_sum(tf.cast(l_true != 0, tf.float32))
    br_size = tf.reduce_sum(tf.cast(r_true != 0, tf.float32))

    tl_ek = tf.reduce_sum(tl) / (tl_size + 1e-4)
    br_ek = tf.reduce_sum(br) / (br_size + 1e-4)
    # ek = tl_ek + br_ek

    pull_tl = tf.reduce_sum(tf.square(tl - tl_ek)) / (tl_size + 1e-4)
    pull_br = tf.reduce_sum(tf.square(br - br_ek)) / (br_size + 1e-4)
    pull = pull_tl + pull_br
    
    push = tf.reduce_sum(tf.maximum(0.0, 1 - tf.abs(tl_ek - br_ek)))
    loss = pull + push
    return loss

def off_loss(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    loss = tf.where(diff < 1, 0.5 * tf.square(diff), diff - 0.5)
    return tf.reduce_mean(loss)