import tensorflow as tf
import numpy as np
import os

# Expected: X is img from dir and y is txt file from dir with same file name

# So input: list of names as X. y is boolean.

class DataGen(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch, im_path, lab_path = None, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = y
        self.im_path = im_path
        self.lab_path = lab_path
        self.batch = batch
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))
    
    def on_epoch_end(self):
        indices = np.arange(len(self.X))
        np.random.shuffle(indices)
        self.X = [self.X[i] for i in indices]
    
    def __getitem__(self, indices):
        X_items = self.X[indices * self.batch: (indices + 1) * self.batch]
        ims = [self.preprocess(os.path.join(self.im_path, i)) for i in X_items]
        X_ims = tf.stack(ims)

        # read files. Make filter for bounding boxes (i.e., same coord space but > 1 coord - just take first case in this instance - 1 bounding box)
        if self.y is True:
            # one bounding box, 7x7 split
            matrix = np.zeros((7, 7, 6))
            for i in X_items:
                name = os.path.splitext(i)[0]
                with open(os.path.join(self.lab_path, f'{name}.txt'), 'r') as text:
                    for p in text:
                        temp = p.strip().split(' ')
                        x_box = int(float(temp[1]) * 7)
                        y_box = int(float(temp[2])  * 7)

                        if matrix[x_box, y_box, 0] == 0:
                            matrix[x_box, y_box, 1: 5] = np.array(temp[1:], dtype = float)
                            matrix[x_box, y_box, 0] = 1
                            matrix[x_box, y_box, 5] = 1
            y_tensor = tf.convert_to_tensor(matrix[np.newaxis, ...])
            return X_ims, y_tensor
        
        return X_ims

    def preprocess(self, file_path):
        im = tf.io.read_file(file_path)
        im = tf.image.decode_image(im, 3)
        im = tf.image.resize(im, (448, 448))
        im = tf.cast(im, tf.float32) / 255
        return im