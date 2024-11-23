import tensorflow as tf
import os
import numpy as np
import cluster
import wn_tree

# bounding box prediction need 13x13x (5+20) * 5 (5 boxes and 5 datapoints and 20 classes - this is variable to needs). Zeros where iou doesn't meet threshold

# Classification: predefined object location. 1 class traverse upwards to object. (n, ) shape

# Detection: anchor boxes, iou for 13x13 split. Multi input - on_epoch_end counter

class DataGen(tf.keras.utils.Sequence):
    # Init with None detection since we can solo fine_tune. Classification dims to 224, 224 or 448, 448 for finetuning
    def __init__(self, c_img_path, c_label_path, classification_dims = (224, 224), det_img_path = None, det_label_path = None, batch = 16, **kwargs):
        super().__init__(**kwargs)
        # paths to directory
        self.class_im_path = c_img_path
        self.class_label_path = c_label_path
        self.classification_dims = classification_dims
        self.det_im_path = det_img_path
        self.det_label_path = det_label_path

        # To get index of each word and get predcessors (above hiarachy) words.
        self.obj_idx, self.predecessors = wn_tree.looper()

        self.batch = batch
        self.counter = 0
        self.det_dims = (448, 448)
        # Generate list of anchor dims
        self.box_dims = cluster.kmeans(det_img_path, det_label_path, 5, random_state = 42)

        self.class_im = np.array(os.listdir(c_img_path))
        self.det_im = np.array(os.listdir(det_img_path))

        self.X = np.random.shuffle(np.concatenate([self.class_im + self.det_im]))
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))

    def on_epoch_end(self):
        self.counter += 1
        if self.counter % 10 == 0:
            choice = np.random.choice(np.arange(288, 608, 32))
            self.det_dims = tuple(choice, choice)
        indices = len(self.X)
        np.random.shuffle(indices)
        self.X = [self.X[i] for i in indices]

    def __getitem__(self, index):
        items = self.X[index * self.batch: (index + 1) * self.batch]
        class_items = self.class_im[np.isin(self.class_im, items)]
        det_items = items[np.isin(items, class_items) == False]

        class_ims = []
        class_labels = []
        det_ims = []
        det_labels = []

        for class_val in class_items:
            temp_im = self.open_img(os.path.join(self.class_im_path, class_val), 'classification')
            temp_tensor = tf.zeros((len(self.obj_idx), ))
            # For ease, say y is stored in a txt file with 1 line as class name (for example "dog")
            file = os.path.join(self.class_label_path, class_val.split(".")[0] + ".txt")
            with open (file, 'r') as line:
                # Get (n,) shape vector wherein 1 class for prior related classes, i.e., word "Husky" is gt = 1 and also "dog", which this traverses upwards. Keep constant value with this.
                temp_locs = [self.obj_idx[i] for i in self.predecessors[line.read_line()]]
                temp_tensor[temp_locs] = 1
                class_labels.append(temp_tensor)
            class_ims.append(temp_im)
        class_ims = tf.convert_to_tensor(class_ims)
        class_labels = tf.stack(class_labels)

        for det_item in det_items:
            temp_im = self.open_img(os.path.join(self.det_im_path, det_item), 'detection')
            lines = os.path.join(self.det_im_path, det_item.split(".")[0] + ".txt")
            with open(lines, 'r') as line:
                # t, x, y, w, h
                temp = [list(map(float, i.strip().split(' '))) for i in line]
            temp_det = anchor_box(self.box_dims, temp, self.det_dims[0])
            det_labels.extend(temp_det)
            det_ims.extend([temp_im for _ in range(len(temp_det))])

        det_ims = tf.convert_to_tensor(det_ims)
        det_labels = tf.stack(det_labels)

        return [class_ims, det_ims], [class_labels, det_labels]

    def open_img(self, path, category):
        im = tf.io.decode_image(path, 'rgb')
        if category == 'classification':
            im = tf.reshape(im, self.classification_dims) / 255.0
        else:
            im = tf.reshape(im, self.det_dims) / 255.0
        return im
    
# Let's say all in % format (yolo format) for simplicity

# 125 filters. 5x5 and 20 classes. (20 + 5) * 5, where 20 + 5 is 20 probs and 5 for coords for object
def anchor_box(box_dims, gt_dims, feature_map_dim):
    strides = 1 / (feature_map_dim / 32)
    tensors = []
    for instance in gt_dims:
        temp_tensor = tf.zeros((feature_map_dim / 32, feature_map_dim / 32, 125))
        for dimension in box_dims:
            for col in range(strides / 2, 1, strides):
                for row in range(strides / 2, 1, strides):
                    iou_val = iou(instance[1:], dimension, (col, row))
                    if iou_val > 0.5:
                        temp_tensor[int(col), int(row), 0] = sigmoid(instance[1] - col) + col  
                        temp_tensor[int(col), int(row), 1] = sigmoid(instance[2] - row) + row
                        temp_tensor[int(col), int(row), 2] = dimension[0] * np.exp(np.log10(instance[3]))
                        temp_tensor[int(col), int(row), 3] = dimension[1] * np.exp(np.log10(instance[4]))
                        temp_tensor[int(col), int(row), 4] = sigmoid(iou_val)
                        # Multiclass afterwards where 0: 5 are coords + obj probs. Next 20 or x are classes. Here we assume multiclass input.
                        temp_tensor[int(col), int(row), int(instance[0]) + 5] = 1
                    elif iou_val < 0.3:
                        # Zeros
                        temp_tensor[int(col), int(row), 0] = sigmoid(instance[1] - col) + col  
                        temp_tensor[int(col), int(row), 1] = sigmoid(instance[2] - row) + row
                        temp_tensor[int(col), int(row), 2] = dimension[0] * np.exp(np.log10(instance[3]))
                        temp_tensor[int(col), int(row), 3] = dimension[1] * np.exp(np.log10(instance[4]))
        tensors.append(temp_tensor)
    return tf.concat(tensors, axis=-1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def iou(gt, ab_dim, ab_xy):
    gt_xmin = gt[0] - gt[2] / 2
    gt_xmax = gt[0] + gt[2] / 2
    gt_ymin = gt[1] - gt[3] / 2
    gt_ymax = gt[1] + gt[3] / 2

    ab_xmin = ab_xy[0] - ab_dim[0] / 2
    ab_xmax = ab_xy[0] + ab_dim[0] / 2
    ab_ymin = ab_xy[1] - ab_dim[1] / 2
    ab_ymax = ab_xy[1] + ab_dim[1] / 2

    union = (min(gt_xmax, ab_xmax) - max(gt_xmin, ab_xmin)) * (min(gt_ymax, ab_ymax) - max(gt_ymin, ab_ymin))
    
    intersection = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin) + (ab_xmax - ab_xmin) * (ab_ymax - ab_ymin) - union

    return union / intersection