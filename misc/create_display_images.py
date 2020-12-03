import numpy as np
import cv2

from scipy.misc import imread

exp_numbers = [145, 285, 286, 287]

files = [f"Z:/Jeffrey-Ede/models/recurrent_conv-1/{n}/input0.tif" for n in exp_numbers]

for i, f in enumerate(files):
    raw = imread(f, "F")

    raw_mask = raw != 0.5

    r = 255*np.ones([96, 96])
    r[raw_mask] = 0
    g = 255*np.ones([96, 96])
    #g[raw_mask] = 0
    b = 255*np.ones([96, 96])
    #b[raw_mask] = 255

    canvas = np.zeros([98,98,3])

    img = np.stack([b, g, r], axis=-1)

    canvas[1:-1,1:-1,:] = img

    img = np.uint8(canvas)

    cv2.imwrite(f"Z:/Jeffrey-Ede/models/recurrent_conv-1/display_{i}.png", img)