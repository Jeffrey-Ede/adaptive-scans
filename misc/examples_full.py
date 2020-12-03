import numpy as np
from scipy.misc import imread
from scipy.stats import entropy

import matplotlib as mpl
#mpl.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['savefig.dpi'] = 300
fontsize = 11
mpl.rcParams['axes.titlesize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize

import matplotlib.mlab as mlab

import cv2

from PIL import Image
from PIL import ImageDraw

columns = 6
rows = 8

parent = "Z:/Jeffrey-Ede/models/recurrent_conv-1/280/"
prependings = ["final_input", "final_truth", "final_generation", "final_truth", "final_generation", "final_truth", "final_generation"]

image_nums = [00+i for i in range(2*rows)]

imgs = []
for i in image_nums:
    for j, prepending in enumerate(prependings[:3]):

        filename = parent + prepending + f"{i}.tif"
        img = imread(filename, mode="F")

        imgs.append(img)

x_titles = [
    "Partial Scan",
    "Target Output",  
    "Generated Output",
    "Partial Scan",
    "Target Output",  
    "Generated Output"
    ]

def scale0to1(img):
    
    min = np.min(img)
    max = np.max(img)

    print(min, max)

    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def block_resize(img, new_size):

    x = np.zeros(new_size)
    
    dx = int(new_size[0]/img.shape[0])
    dy = int(new_size[1]/img.shape[1])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            px = img[i,j]

            for u in range(dx):
                for v in range(dy):
                    x[i*dx+u, j*dy+v] = px

    return x
    
#Width as measured in inkscape
scale = 4
width = scale * 2.2
height = 2*1.15*scale* (width / 1.618) / 2.2 / 1.96


w = h = 224

subplot_cropsize = 64
subplot_prop_of_size = 0.625
subplot_side = int(subplot_prop_of_size*w)
subplot_prop_outside = 0.25
out_len = int(subplot_prop_outside*subplot_side)
side = w+out_len

print(imgs[1])

f=plt.figure(figsize=(rows, columns))

for i in range(rows):
    for j in range(1, columns+1):
        
        img = imgs[columns*i+j-1]
        k = i*columns+j

        ax = f.add_subplot(rows, columns, k)
        plt.imshow(img, cmap='gray', norm=mpl.colors.Normalize(vmin=0.,vmax=1.))
        plt.xticks([])
        plt.yticks([])

        ax.set_frame_on(False)
        if not i:
            ax.set_title(x_titles[j-1])

f.subplots_adjust(wspace=-0.01, hspace=0.04)
f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)

f.set_size_inches(width, height)

#plt.show()
f.savefig(parent+'examples_full-1.png', bbox_inches='tight')

