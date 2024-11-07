import tensorflow as tf
import numpy as np
import ss
import os
import cv2

class BoxDataGen(tf.keras.utils.Sequence):
    def __init__(self, X_path, y_path, batch, **kwargs):
        super().__init__(**kwargs)
        self.xpath = X_path
        self.ypath = y_path
        self.batch = batch
        self.X = os.listdir(X_path)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))
    
    def on_epoch_end(self):
        lengths = np.arange(len(self.X))
        self.X = [self.X[i] for i in lengths]

    def __getitem__(self, indices):
        items = self.X[indices * self.batch: (indices + 1) * self.batch]
        bb = []
        bb_gt = []
        for i in items:
            im_path = os.path.join(self.xpath, i)
            _, boxes = ss.selective_search(im_path)
            # Make only 1 gt for simplicity. Would have to call and make loop/ vectorize if implementing multi
            if self.ypath is not None:
                with open(os.path.join(self.ypath, i.split('.')[0] + '.txt')) as text:
                    # gt in form t, x, y, w, h
                    gt = text.readline().split(' ')
                    dims = [float(i) * 224 for i in gt[1:]]
            for box in boxes:
                im = cv2.imread(im_path)[box[1]: box[1] + box[3], box[0]: box[0] + box[2]]
                if self.ypath is not None:
                    temp = ss.iou(box, dims)
                    if temp > 0.5:
                        bb.append(self.image_pre(im))
                        bb_gt.append(int(gt[0]))
                    else:
                        bb.append(self.image_pre(im))
                        bb_gt.append(0)
                else:
                    bb.append(self.image_pre(im))
        if self.ypath is not None:
            return tf.stack(bb), tf.stack(bb_gt)
        else:
            return tf.stack(bb), 
        

    def image_pre(self, img):
        im = tf.convert_to_tensor(img)
        im = tf.image.resize(im, (224, 224))
        im = im / 255.0
        return im