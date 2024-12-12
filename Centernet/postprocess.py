import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

'''
Input of image and preds which is nn prediction array
- Let's say input is image path
'''

def post_process(im, preds):
    classes_ = {0: 'creatures', 1: 'fish', 2: 'jellyfish', 3: 'penguin', 4: 'puffin', 5: 'shark', 6: 'starfish', 7: 'stingray'}

    tl_heat, tl_embed, tl_off, br_heat, br_embed, br_off, cen_heat, cen_off = preds[..., 0: 8], preds[..., 8], preds[..., 9: 11], preds[..., 11: 19], preds[..., 19],\
          preds[..., 20: 22], preds[..., 22: 30], preds[..., 30: 32]
    
    tl_pos = _nms(tl_heat)
    br_pos = _nms(br_heat)
    cen_pos = _nms(cen_heat, True)
    
    br_pos[:, 0] = br_pos[:, 0] + br_off[br_pos[:, 0], br_pos[:, 1], 0]
    br_pos[:, 1] = br_pos[:, 1] + br_off[br_pos[:, 0], br_pos[:, 1], 1]
    tl_pos[:, 0] = tl_pos[:, 0] + tl_off[tl_pos[:, 0], tl_pos[:, 1], 0]
    tl_pos[:, 1] = tl_pos[:, 1] + tl_off[tl_pos[:, 0], tl_pos[:, 1], 1]
    cen_pos[:, 0] = cen_pos[:, 0] + cen_off[cen_pos[:, 0], cen_pos[:, 1], 0]
    cen_pos[:, 1] = cen_pos[:, 1] + cen_off[cen_pos[:, 0], cen_pos[:, 1], 1]

    tl_scores = tl_embed[tl_pos[:, 0], tl_pos[:, 1]]
    br_scores = tl_embed[br_pos[:, 0], br_pos[:, 1]]
    pairs = []
    for i, tl in enumerate(tl_pos):
        for j, br in enumerate(br_pos):
            if (abs(tl_embed[tl[0], tl[1]] - br_embed[br[0], br[1]]) <= 0.5) and (tl[2] == br[2]):
                area = (br[0] - tl[0]) * (br[1] - tl[1])
                if area <= 0:
                    continue
                elif area > 150:
                    ctlx = (6 * tl[1] + 4 * br[1]) / 10
                    ctly = (6 * tl[0] + 4 * br[0]) / 10
                    cbrx = (4 * tl[1] + 6 * br[1]) / 10
                    cbry = (4 * tl[0] + 6 * br[0]) / 10
                else:
                    ctlx = (4 * tl[1] + 2 * br[1]) / 10
                    ctly = (4 * tl[0] + 2 * br[0]) / 10
                    cbrx = (2 * tl[1] + 4 * br[1]) / 10
                    cbry = (2 * tl[0] + 4 * br[0]) / 10
                for ceny, cenx, _ in cen_pos:
                    if (cenx <= cbrx and cenx <= ctlx) and (ceny <= cbry and ceny <= ctly):
                        score = (tl_scores[i] + br_scores[j]) / 2
                        pairs.append((tl.tolist(), br.tolist(), score))
                        continue
                    
    plt.figure(figsize = (10, 10))
    im = plt.imread(im)
    plt.imshow(im)
    width, height = im.shape[1], im.shape[0] 
    width_scale = width / 64
    height_scale = height / 64

    for l, r, s in pairs:
        box_width = (r[1] - l[1]) * width_scale
        box_height = (r[0] - l[0]) * height_scale
        rect = patch.Rectangle((l[1] * width_scale, l[0] * height_scale), box_width, box_height, ec = 'green', lw = 2, fill = None)
        plt.gca().add_patch(rect)
        plt.text(l[1] * width_scale, l[0] * height_scale, f'{classes_[l[2]]}: {s}')
    plt.tight_layout()
    plt.show()

def _nms(inputs, center = False):
    inputs = np.pad(inputs, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    windows = np.lib.stride_tricks.sliding_window_view(inputs, (3, 3, 1))
    max_vals = np.max(windows, axis = (3, 4, 5))
    rows, cols, zdims = np.meshgrid(np.arange(64), np.arange(64), np.arange(8), indexing = 'ij')
    nms_array = np.stack((rows, cols, zdims, max_vals), axis = -1).reshape(-1, 4)
    nms_array = sorted(nms_array, key = lambda x: x[3])[-100:]
    if center is True:
        nms_array = [i for i in nms_array if i[3] > 0.3]
    return np.array([[int(x[0]), int(x[1]), int(x[2])] for x in nms_array])