import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat

# Doesn't work yet

# Array input
def im_show(im, b1, b2 = None):
    plt.figure()
    im = plt.imshow(im)
    x = b1[1] - b1[3] / 2
    y = b1[2] - b1[4] / 2
    plt.gca().add_patch(pat.Rectangle((x, y), b1[3], b1[4]))

    if b2 is not None:
        x = b2[1] - b2[3] / 2
        y = b2[2] - b2[4] / 2
        plt.gca().add_patch(pat.Rectangle((x, y), b2[3], b2[4]))
    plt.show()
