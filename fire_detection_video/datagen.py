import tensorflow as tf
import numpy as np
import os

# Awful data gen but works for simple objective

class DataGen(tf.keras.utils.Sequence):
    def __init__(self, f_p, d_p, y_val, batch, training, **kwargs):
        super().__init__(**kwargs)
        self.batch = batch
        self.y_val = y_val
        self.training = training
        
        if training is True:
            fire = os.listdir(f_p)
            default = os.listdir(d_p)

            f = [self.preprocess(os.path.join(f_p, i)) for i in fire]
            d = [self.preprocess(os.path.join(d_p, i)) for i in default]
            
            self.X = tf.stack(f + d)

            indexes = np.arange(len(self.X))
            np.random.shuffle(indexes)
            self.X = tf.gather(self.X, indexes)

            if y_val is True:
                self.y = tf.convert_to_tensor(np.concatenate((np.ones(len(fire)), np.zeros(len(default))), axis=0))
                self.y = tf.gather(self.y, indexes)
        else:
            fire = os.listdir(f_p)
            f = [self.preprocess(os.path.join(f_p, i)) for i in fire]
            self.X = tf.stack(f)

    def __len__(self):
        return int(np.ceil(self.X.get_shape().as_list()[0] / self.batch))
    
    def on_epoch_end(self):
        indexes = np.arange(len(self.X))
        np.random.shuffle(indexes)
        self.X = tf.gather(self.X, indexes)
        if self.y_val is True:
            self.y = tf.gather(self.y, indexes)

    def __getitem__(self, index):
        if self.training is True:
            train_data = self.X[index * self.batch: (index + 1) * self.batch]

            if self.y_val is True:
                return train_data, self.y[index * self.batch: (index + 1) * self.batch]
            
            return train_data
        else:
            return self.X,

    def preprocess(self, file_path):
        im = tf.io.read_file(file_path)
        im = tf.image.decode_image(im, 3)
        im = tf.image.resize(im, (96, 96))
        im = tf.cast(im, tf.float32) / 255
        return im