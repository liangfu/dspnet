#!/usr/bin/env python

import cv2
import sys
import numpy as np

def getpalette(num_cls):
    # this function is to get the colormap for visualizing the segmentation mask
    n = num_cls
    palette = [0]*(n*3)
    for j in xrange(0,n):
        lab = j
        palette[j*3+0] = 0
        palette[j*3+1] = 0
        palette[j*3+2] = 0
        i = 0
        while (lab > 0):
            palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return palette

def color2index(label):
    palette = np.array(getpalette(256)).reshape((256,3))
    _color2index = {tuple(p):idx for idx,p in enumerate(palette)} # (255, 255, 255) : 0,
    h, w, ch = label.shape
    label = map(lambda x:_color2index[tuple(x)],label.reshape((h*w,ch)).tolist())
    label = np.array(label).reshape((h,w))
    return label

def index2color(out_img):
    palette = np.array(getpalette(256)).reshape((256,3))
    _index2color = {idx:p.tolist() for idx,p in enumerate(palette)} # (255, 255, 255) : 0,
    h, w = out_img.shape
    out_img = map(lambda x:_index2color[x[0]],out_img.reshape((h*w,1)).tolist())
    out_img = np.array(out_img).reshape((h,w,3)).astype(np.uint8)
    return out_img


