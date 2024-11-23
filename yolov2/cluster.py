import os
import cv2
import gc
import numpy as np

def kmeans(img_path, label_path, num_clusters, random_state = None):
    gt_dims = []
    ims = os.listdir(img_path)
    for im in ims:
        # (w, h)
        size = cv2.imread(os.path.join(img_path, im)).shape[1::-1]
        with open(os.join(label_path, im.split('.')[0] + '.txt'), 'r') as file:
            for line in file.read_line():
                w = int(line[3])
                h = int(line[4])
                sc_w = w * (size[0] / w)
                sc_h = h * (size[1] / h)
                gt_dims.append(zip(sc_w, sc_h))

        del sc_w, sc_h, w, h, line, file, size
        gc.collect()

    clustoids = []
    if random_state is not None:
        np.random.seed(random_state)
    init_dims = tuple(np.random.rand(2)) * 448
    clustoids.append(init_dims)
    
    # Iter until n clusters found
    for _ in range(num_clusters - 1):
        distance = []

        # Per data loop
        for data in gt_dims:
            max_dist = -1

            for dist in clustoids:
                temp_dist = kmeans_iou(data, dist)
                max_dist = max(temp_dist, max_dist)
            distance.append(max_dist)

        clustoids.append(gt_dims[np.argmax(distance)])
        del distance, temp_dist, max_dist
    
    
    clust_lists = [list(i) for i in clustoids]
    for data_point in gt_dims:
        min_dist = 1
        index = -1
        best = None
        for idx, centroid in enumerate(clustoids):
            val = kmeans(data_point, centroid)
            min_dist = min(min_dist, val)
            if min_dist == val:
                index = idx
            best = centroid
        clust_lists[index].append(best)
    
    # list of clusters of (x, w)
    return [[(sum(p) / len(i), sum(k) / len(i)) for p, k in i] for i in clust_lists]


def kmeans_iou(gt, clust):
    w_gt = gt[0]
    h_gt = gt[1]
    w_cl = clust[0]
    h_cl = clust[1]
    intersection = min(w_gt, w_cl) * min(h_gt, h_cl)
    union = (w_gt * h_gt) + (w_cl * h_cl) - intersection
    return 1 - intersection / union