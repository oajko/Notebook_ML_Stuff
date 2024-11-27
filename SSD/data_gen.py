import tensorflow as tf
import os
import numpy as np

# Input shape of 300

# === For ease ===
# boxes: AR: 1:1, 2:1, 1:2
# Sizes of box of 0.2, 0.4, 0.6
# All imgs are inputted as % in yolo format: class label, x, y, w, h
# All files names are =
# 5 classes

# Let's just say one img per (other ml models have multimodel looping/ handling)

class DataGen(tf.keras.utils.Sequence):
    def __init__(self, xpath, batch, labelpath = None, **kwargs):
        super().__init__(**kwargs)
        self.batch = batch
        self.xpath = xpath
        self.labelpath = labelpath
        self.X = os.listdir(self.xpath)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))

    def on_epoch_end(self):
        index = np.arange(len(self.X))
        np.random.shuffle(index)
        self.X = [self.X[i] for i in index]

    def __getitem__(self, index):
        training = self.X[index * self.batch: (index + 1) * self.batch]
        default_ims = []
        default_regr = []
        default_class = []

        for img in training:
            im = os.path.join(self.xpath, img)
            im = self.load_img(im)
            default_ims.append(im)

            if self.labelpath is not None:
                label_ = os.path.join(self.labelpath, img.split(".")[0] + ".txt")
                with open (label_, 'r') as line:
                    temp_line = line.read_line().split(" ")
                temp_line[1:] = [float(i) for i in temp_line[1:]]
                temp_line[0] = int(temp_line[0])
                regr_labels, class_labels = default_box(temp_line)
                default_regr.append(regr_labels)
                default_class.append(class_labels)

        images = tf.stack(default_ims)
        if self.labelpath is not None:
            return (images, [tf.stack(default_regr), tf.stack(default_class)])
        return images
    
    def load_img(self, path):
        im = tf.io.decode_image(path, 'rgb')
        im = tf.reshape(im, (300, 300)) / 255.0
        return im
    
def default_box(gt):
    counter = 0
    conv_dims = [37, 18, 9, 5, 3, 2, 1]
    sizes = [0.2, 0.4, 0.6]
    aspect_ratio = [(1, 1), (1, 2), (2, 1)]

    flattened_agg = []
    conf_agg = []
    for cov_dim in conv_dims:
        tensor = tf.zeros((cov_dim, cov_dim, 9 * 4))
        conf_tensor = tf.zeros((cov_dim, cov_dim, 9 * 6))
        pos_score = 0
        neg_arr = []
        for ar in aspect_ratio:
            for size in sizes:
                scale = dim_scale(cov_dim)
                asp = (ar[0] * size[0]) / (ar[1] * size[1])
                width = scale * np.sqrt(asp)
                height = scale / np.sqrt(asp)
                for row in range(0, cov_dim):
                    for col in range(0, cov_dim):
                        norm_row = (row + 0.5)/ cov_dim
                        norm_col = (col + 0.5)/ cov_dim
                        iou_score = iou(gt[1:], (norm_row, norm_col, width, height))
                        if iou_score > 0.5:
                            x_diff = gt[1] - norm_col
                            y_diff = gt[2] - norm_row
                            w_diff = np.log(gt[3])
                            h_diff = np.log(gt[4])
                            tensor[row, col, counter * 4: (counter + 1) * 4] = [x_diff, y_diff, w_diff, h_diff]
                            conf_tensor[row, col, gt[0] + 1 + counter * 5] = 1
                            pos_score += 1
                        else:
                            neg_arr.append((iou_score, [row, col, counter * 5]))
                counter += 1

        # Covers for hard negatives. We assign 1 class to hard negatives on index = 0 on z scale for each anchor box. Believe that's how the paper does it. Rest are just zeros
        neg_num = pos_score * 3
        neg_arr = sorted(neg_arr, reverse = True)[:neg_num]
        negs = np.array([j for i, j in neg_arr])
        for row, col, anchor_idx in negs:
            conf_tensor[row, col, anchor_idx * 5] = 1

        flattened_agg.extend(tf.reshape(tensor, (-1, )))
        conf_agg.extend(tf.reshape(conf_tensor, (-1, )))

    return tf.concat(flattened_agg, axis = 1), tf.concat(conf_agg, axis = 1)
# min = 0.2, max = 0.6. We use above for scaling. Paper uses 0.9 to 0.2, but it shouldn't really matter
def dim_scale(scale):
    return scale * (scale - 0.1) / 4 * 8

def iou(gt, preds):
    gt_xmin = gt[0] - gt[2] / 2
    gt_xmax = gt[0] + gt[2] / 2
    gt_ymin = gt[1] - gt[3] / 2
    gt_ymax = gt[1] + gt[3] / 2

    pred_xmin = preds[0] - preds[2] / 2
    pred_xmax = preds[0] + preds[2] / 2
    pred_ymin = preds[1] - preds[3] / 2
    pred_ymax = preds[1] + preds[3] / 2

    intersection = (min(gt_xmax, pred_xmax) - max(gt_xmin, pred_xmin)) * (min(gt_ymax, pred_ymax) - max(gt_ymin, pred_ymin))

    union = (gt_xmax - gt_xmin) * (gt_ymax - gt_xmin) + (pred_xmax - pred_xmin) * (pred_ymax - pred_xmin) - intersection
    return intersection / union