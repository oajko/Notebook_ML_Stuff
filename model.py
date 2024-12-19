import tensorflow as tf
import numpy as np

'''
Preprocessing is done in DataGen I suppose.
'''

def model():
    input_layer = tf.keras.layers.Input((512, 512))
    output_layer = tf.keras.layers.Input((512, 512))
    transformer = Transformer()([input_layer, output_layer])
    return tf.keras.Model([input_layer, output_layer], transformer)

class Transformer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.posinp = PositionEncoding()
        self.posout = PositionEncoding()
        self.enc1 = Encoder()
        self.dec1 = Decoder()
    
    def call(self, inputs):
        inputs_, outputs_ = inputs
        inputs_ = self.posinp(inputs_)
        outputs_ = self.posout(outputs_)
        x = self.enc1(inputs_)
        x = self.dec1([x, outputs_])
        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.multihead = MultiheadAttention(8)
        self.ff = FreeForward()
    
    def call(self, inputs):
        self_attention = self.multihead([inputs, inputs, inputs])
        ff = self.ff(self_attention)
        return ff

class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.maskattention = MultiheadAttention(8, True)
        self.multihead = MultiheadAttention(8)
        self.ff = FreeForward()
    
    def call(self, inputs):
        inputs_, outputs_ = inputs
        mask_att = self.maskattention([outputs_, outputs_, outputs_])
        norm_att = self.multihead([inputs_, inputs_, mask_att])
        ff = self.ff(norm_att)
        return ff

class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, heads, mask = False, **kwargs):
        super(MultiheadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.mask = mask
        self.q_lin = tf.keras.layers.Dense(int(512 / heads), activation = 'linear')
        self.k_lin = tf.keras.layers.Dense(int(512 / heads), activation = 'linear')
        self.v_lin = tf.keras.layers.Dense(int(512 / heads), activation = 'linear')
        self.fin_lin = tf.keras.layers.Dense(512, activation = 'linear')
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        q, k, v = inputs
        if self.mask is True:
            meshx, meshy = tf.meshgrid(np.arange(1, 513), np.arange(1, 513))
            meshy = tf.transpose(meshy, [1, 0])
            mask = tf.where(meshy >= meshx, 1.0, 0.0)
            q = q * mask
            k = k * mask
            v = v * mask
        q = self._reshape(self.q_lin(q), False)
        k = self._reshape(self.k_lin(k), False)
        v = self._reshape(self.v_lin(v), False)

        dotprod = self._dotproduct(q, k)
        norm = self._norm(dotprod, int(512 / self.heads))
        dotprod2 = tf.matmul(norm, v)
        concat = self._reshape(dotprod2, True)
        fin_lin = self.fin_lin(concat)
        return self.norm(fin_lin) + inputs[0]

    def _reshape(self, inputs, dim2):
        if dim2 is False:
            return tf.reshape(inputs, (self.heads, 512, int(512 / self.heads)))
        return tf.reshape(inputs, (512, 512))
    
    def _dotproduct(self, q, k):
        return tf.matmul(q, k, transpose_b = True)

    def _norm(self, dotprod, dim):
        dim_tensor = tf.cast(dim, tf.float32)
        return dotprod / tf.sqrt(dim_tensor)

class FreeForward(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FreeForward, self).__init__(**kwargs)
        self.updense = tf.keras.layers.Dense(2048, activation = 'relu')
        self.downdense = tf.keras.layers.Dense(512, activation = 'relu')
        self.norm = tf.keras.layers.LayerNormalization()
    
    def call(self, inputs):
        x = self.updense(inputs)
        x = self.downdense(x)
        return self.norm(x) + inputs

class PositionEncoding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PositionEncoding, self).__init__(**kwargs)
    
    def call(self, inputs):
        position = np.zeros((512, 512))
        for row in range(512):
            for col in range(256):
                position[row, col * 2] = np.sin(row / np.power(10000, (2 * col) / 512))
                position[row, col * 2 + 1] = np.cos(row / np.power(10000, (2 * col) / 512))
        return inputs + tf.convert_to_tensor(position, tf.float32)