import covnet
import os
import ss
import cv2
import numpy as np
from sklearn.svm import SVC
import pickle
from sklearn.linear_model import LinearRegression

def covnet_preds(x_path, y_path, cnn_path):
    x_outputs = []
    y_preds = []
    dim_preds = []
    dim_act = []
    x = os.listdir(x_path)
    model = covnet.cnn_features(train = False, path = cnn_path)
    for i in x:
        path = os.path.join(x_path, i)
        with open(os.path.join(y_path, i.split('.')[0] + '.txt')) as text:
            gt = text.readline().split(' ')
            dims = [float(i) * 224 for i in gt[1:]]
        _, boxes = ss.selective_search(path)
        for box in boxes:
            temp = ss.iou(box, dims)
            im = cv2.imread(path)[box[1]: box[1] + box[3], box[0]: box[0] + box[2]]
            resize_im = cv2.resize(im, (224, 224))
            resize_im = np.expand_dims(resize_im, axis = 0)
            x_outputs.append(model.predict(resize_im))
            dim_preds.append(box)
            dim_act.append(dims)
            if temp > 0.5:
                y_preds.append(1)
            else:
                y_preds.append(0)

    return x_outputs, y_preds, dim_act, dim_preds


def SVM(inputs, outputs, train, save = None):
    if train is True:
        inputs = [i.flatten() for i in inputs]
        clf = SVC().fit(inputs, outputs)
        if save is not None:
            with open(save, 'wb') as file:
                pickle.dump(clf, file)
    else:
        with open(save, 'rb') as file:
            clf = pickle.load(file)
    return clf

# list of [x, y, w, h]
def lin_regr(inputs, outputs_true, outputs_preds, train, save = None):
    if train is True:
        diff = []
        for i, j in zip(outputs_true, outputs_preds):
            diffx = i[0] - (j[0] - j[2] / 2)
            diffy = i[1] - (j[1] + j[3] / 2)
            diffw = i[2] - j[2]
            diffh = i[3] - j[3]

            diff.append(np.hstack([diffx, diffy, diffw, diffh]))
        inputs = [i.flatten() for i in inputs]
        clf = LinearRegression().fit(inputs, diff)
        if save is not None:
            with open(save, 'wb') as file:
                pickle.dump(clf, file)
    else:
        with open(save, 'rb') as file:
            clf = pickle.load(file)
    return clf