import tensorflow as tf
import numpy as np
import os
import re

# For simplicity, we assume image ends in _num

class FineTuneDataGen(tf.keras.utils.Sequence):
    def __init__(self, X_path, batch, train, **kwargs):
        super().__init__(**kwargs)
        self.xpath = X_path
        self.batch = batch
        self.train = train
        self.X = os.listdir(X_path)
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))

    def on_epoch_end(self):
        index = np.arange(len(self.X))
        np.random.shuffle(index)
        self.X = [self.X[i] for i in index]

    def __getitem__(self, indices):
        items = self.X[indices * self.batch: (indices + 1) * self.batch]
        ims = []
        if self.train is True:
            ys = []
            for img in items:
                ims.append(self.img_clean(os.path.join(self.xpath, img)))
                ys.append(re.search(r"_(\d).", img).group(1))
            return tf.stack(ims), tf.stack(ys)
        
        for img in items:
            ims.append(self.img_clean(os.path.join(self.xpath, img)))
        return tf.stack(ims),

    def img_clean(self, img):
        im = tf.io.decode_image(img, 'rgb')
        im = tf.image.resize(im, (224, 224)) / 255.0
        return im
    

# Input raw image. Output in anchor box form. Class probas and regression
# shape is 4 * 9 * 14
# 224x224 img

# Path to img. Then yolo format
class RPNDataGen(tf.keras.utils.Sequence):
    def __init__(self, xpath, ypath, batch, train, **kwargs):
        super().__init__(**kwargs)
        self.xpath = xpath
        self.ypath = ypath
        self.batch = batch
        self.train = train
        self.X = os.listdir(xpath)
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))
    
    def on_epoch_end(self):
        index = np.arange(len(self.X))
        np.random.shuffle(index)
        self.X = [self.X[i] for i in index]

    def __getitem__(self, indices):
        items = self.X[indices * self.batch: (indices + 1) * self.batch]
        x_tens = []
        y_tens = []
        x_gt = []

        for img in items:
            im = self.img_clean(os.path.join(self.xpath, img))
            if self.train is True:
                y_locs = []
                with open (os.path.join(self.ypath, img.split('.'[0]) + '.txt'), 'r') as file:
                    temp_files = file.readlines()
                    temp_files = [i.split(' ') for i in temp_files]
                    temp_files = [(int(item[0]), list(map(float, item[1:]))) for item in temp_files]
                    y_locs.append(temp_files)
                gt_anchors = anchor_box(im, y_locs)
                for anchor, class_preds in gt_anchors:
                    x_tens.append(im)
                    y_tens.append(anchor)
                    x_gt.append(class_preds)
            else:
                x_tens.append(im)

        if self.train is True:
            return x_tens, [tf.stack(y_tens), tf.stack(x_gt)]
        return tf.stack(x_tens)
    
    def img_clean(self, img):
        im = tf.io.decode_image(img, 'rgb')
        im = tf.image.resize(im, (224, 224)) / 255.0
        return im
    
# Image input as tensor
def anchor_box(img, gts):
    # Should be 224, 224
    row = tf.get_shape(img)[0]
    col = tf.get_shape(img)[1]
    sizes = [128, 256, 512]
    aspect_ratios = [(1, 1), (2, 1), (1, 2)]

    gt_list = []
    class_gt = []

    for gt in gts:
        temp_tensor = tf.zeros([14, 14, 9 * 4])
        class_tensor = tf.zeros([14, 14, 9 * 2])
        num = 0
        for size in sizes:
            for ratio in aspect_ratios:
                for i in range(8, row, 16):
                    for j in range(8, col, 16):
                        
                        grid_i = i // 16
                        grid_j = j // 16
                        
                        # Get percentages
                        x = i / 224
                        y = j / 224
                        w = (size * ratio[0]) / 224
                        h = (size * ratio[1]) / 224
                        iou_val = iou((x, y, w, h), gt[1:])

                        if iou_val > 0.7 or iou_val < 0.3:
                            c_x = (gt[1] - x) / w
                            c_y = (gt[2] - y) / h
                            c_w = np.log(gt[3] / w)
                            c_h = np.log(gt[4] / h)
                            temp_tensor[grid_i, grid_j, num * 4: num * 4 + 4] = c_x, c_y, c_w, c_h
                            if iou_val > 0.7:
                                class_tensor[grid_i, grid_j, num * 2: num * 2 + 4] = 1, 0
                            else:
                                class_tensor[grid_i, grid_j, num * 2: num * 2 + 4] = 0, 1
                num += 1
        gt_list.append(temp_tensor)
        class_gt.append(temp_tensor)

    # Stack on batches
    return gt_list, class_gt

# All in min_max form.
def iou(x, y):
    gtx_min = max(y[0] - y[2] / 2, 0)
    gtx_max = min(y[0] + y[2] / 2, 1)
    gty_min = max(y[1] - y[3] / 2, 0)
    gty_max = min(y[1] + y[3] / 2, 1)

    abx_min = max(x[0] - x[2] / 2, 0)
    abx_max = min(x[0] + x[2] / 2, 1)
    aby_min = max(x[1] - x[3] / 2, 0)
    aby_max = min(x[1] + x[3] / 2, 1)

    width = min(gtx_max, abx_max) - max(gtx_min, abx_min)
    height = min(gty_max, aby_max) - max(gty_min, aby_min)

    intersection = width * height

    b1 = (gtx_max - gtx_min) * (gty_max - gty_min)
    b2 = (abx_max - abx_min) * (aby_max - aby_min)

    union = b1 + b2 - intersection

    return intersection / union