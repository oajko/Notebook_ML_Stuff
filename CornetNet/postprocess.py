import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

'''
Output is a 128x128x22 tensor (concatenated)
- Resize to original image scale and apply NMS
'''

def prediction_display(im, preds):
    classes_ = {0: 'creatures', 1: 'fish', 2: 'jellyfish', 3: 'penguin', 4: 'puffin', 5: 'shark', 6: 'starfish', 7: 'stingray'}
    
    heatl, embedl, offl, heatr, embedr, offr = preds[..., 0: 9], preds[..., 9], preds[..., 10: 12], preds[..., 12: 20], preds[..., 20], preds[..., 21: 23] 
    heatl = np.pad(heatl, ((0, 2), (0, 2), (0, 0)), mode = 'constant')
    heatr = np.pad(heatr, ((0, 2), (0, 2), (0, 0)), mode = 'constant')
    # Corner locations: x, y, z
    tl_pos = np.array(nms(heatl))
    br_pos = np.array(nms(heatr))

    br_pos[:, 0] += offl[br_pos[:, 0], br_pos[:, 1], 0]
    br_pos[:, 1] += offl[br_pos[:, 0], br_pos[:, 1], 1]
    tl_pos[:, 0] += offr[tl_pos[:, 0], tl_pos[:, 1], 0]
    tl_pos[:, 1] += offr[tl_pos[:, 0], tl_pos[:, 1], 1]

    distances = np.abs(embedl[:, np.newaxis] - embedr).sum(axis = 2)
    tl_scores = heatr[tl_pos[:, 0], tl_pos[:, 1], tl_pos[:, 2]]
    br_scores = heatl[br_pos[:, 0], br_pos[:, 1], br_pos[:, 2]]

    pairs = []
    for i, tl in enumerate(tl_pos):
        for j, br in enumerate(br_pos):
            if distances[i, j] <= 0.5 and tl[2] == br[2]:
                score = (tl_scores[i] + br_scores[j]) / 2
                pairs.append((tl.tolist(), br.tolist(), score))
    
    plt.figure(figsize = (10, 10))
    plt.imshow(im)
    width, height = plt.gcf().get_size_inches()
    width_scale = width / 128
    height_scale = height / 128

    for l, r, s in pairs:
        box_width = (l[1] - r[1]) * width_scale
        box_height = (l[0] - r[0]) * height_scale
        rect = patch.Rectangle((l[1] * width_scale, l[0] * height_scale), box_width, box_height, ec = 'green', lw = 2, fill = None)
        plt.gca().add_patch(rect)
        plt.text(l[1] * width_scale, l[0] * height_scale, f'{classes_[l[2]]}: {s}')
    plt.tight_layout()
    plt.show()


def nms(heatmap):
    window = np.lib.stride_tricks.sliding_window_view(heatmap, (3, 3, 1))
    max_vals = np.max(window, axis = (3, 4))
    rows, cols, zdims = np.meshgrid(np.arange(128), np.arange(128), np.arange(8), indexing = 'ij')
    nms_array = np.stack((rows, cols, zdims, max_vals), axis = -1).reshape(-1, 4).to_list()
    nms_array = sorted(nms_array, key = lambda x: x[3])[-100:]
    return [[int(x[0]), int(x[1]), int(x[2])] for x in nms_array]
    # return nms_array