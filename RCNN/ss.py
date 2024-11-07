import cv2
import selectivesearch
import matplotlib.pyplot as plt

def selective_search(path):
    im = cv2.imread(path)

    im = cv2.resize(im, (224, 224))
    label, regions = selectivesearch.selective_search(im, scale = 200, sigma = 0.5, min_size = 200)

    # Avoid duplicates
    checker = set()

    for i in regions:
        if i['rect'] in checker:
            continue
        checker.add(i['rect'])
    
    return list(label), list(checker)

# Would have to do for loop if multiple gt Or for loop via datagen
# gt and pred input as x, y, w, h.
def iou(pred, gt):
    # x, y, w, h
    p_xmin = pred[0]
    p_ymax = pred[1]
    p_xmax = pred[0] + pred[2]
    p_ymin = pred[1] + pred[3]

    g_xmin = gt[0] - gt[2] / 2
    g_ymax = gt[1] + gt[3] / 2
    g_xmax = gt[0] + gt[2] / 2
    g_ymin = gt[1] - gt[3] / 2

    i1 = max(0, min(p_xmax, g_xmax) - max(p_xmin, g_xmin))
    i2 = max(0, min(p_ymax, g_ymax) - max(p_ymin, g_ymin))

    intersection = i1 * i2

    b1 = (p_xmax - p_xmin) * (p_ymax - p_ymin)
    b2 = (g_xmax - g_xmin) * (g_ymax - g_ymin)

    union = b1 + b2 - intersection

    return intersection / union

def ss_display(path):
    im = cv2.imread(path)
    _, regions = selectivesearch.selective_search(im, scale = 200, sigma = 0.5, min_size = 200)
    checker = set()
    for i in regions:
        if i['rect'] in checker:
            continue
        checker.add(i['rect'])
    im_copy = im.copy()
    for (x, y, w, h) in checker:
        cv2.rectangle(im_copy, (x, y), (x + w, y + h), (0, 255, 0), 1)
    fig, axes = plt.subplots(1, 2, sharex = True, sharey = True, figsize = (12, 4))
    axes[0].imshow(im)
    axes[1].imshow(im_copy)
    plt.show()
