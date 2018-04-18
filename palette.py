#!/usr/bin/env python

import cv2
import sys
import numpy as np

def get_palette(num_cls):
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
    palette = np.array(get_palette(256)).reshape((256,3))
    # from dataset.cs_labels import labels
    # lut = np.zeros((256,3))
    # for l in labels:
    #     if l.trainId<255 and l.trainId>=0:
    #         lut[l.trainId,:]=list(l.color)
    # palette = np.array(lut).reshape((256,3))
    _color2index = {tuple(p):idx for idx,p in enumerate(palette)} # (255, 255, 255) : 0,
    h, w, ch = label.shape
    label = map(lambda x:_color2index[tuple(x)],label.reshape((h*w,ch)).tolist())
    label = np.array(label).reshape((h,w))
    return label

def index2color(out_img):
    # palette = np.array(get_palette(256)).reshape((256,3))
    from dataset.cs_labels import labels
    lut = np.zeros((256,3))
    for l in labels:
        if l.trainId<255 and l.trainId>=0:
            lut[l.trainId,:]=list([l.color[2],l.color[1],l.color[0]])
    palette = lut
    lut_r = lut[:,0].astype(np.uint8)
    lut_g = lut[:,1].astype(np.uint8)
    lut_b = lut[:,2].astype(np.uint8)
    out_img_r = cv2.LUT(out_img.astype(np.uint8),lut_r)
    out_img_g = cv2.LUT(out_img.astype(np.uint8),lut_g)
    out_img_b = cv2.LUT(out_img.astype(np.uint8),lut_b)
    out_img = cv2.merge((out_img_r,out_img_g,out_img_b))
    return out_img


