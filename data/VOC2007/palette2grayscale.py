#!/usr/bin/env python

"""
Usage Example:
  ./palette2grayscale.py 000001.png 000001_index.png
"""

import Image
import cv2
import numpy as np
from utils import getpalette
import os,sys

# palette=getpalette(256)
palette = np.array(getpalette(256)).reshape((256,3))
color2index = {tuple(p):idx for idx,p in enumerate(palette)} # (255, 255, 255) : 0,
index2color = {idx:p.tolist() for idx,p in enumerate(palette)} # (255, 255, 255) : 0,

# label = cv2.imread(sys.argv[1])
# h, w, ch = label.shape
# print label.reshape((h*w,ch)).tolist()
# label = map(lambda x:color2index[tuple(x)],label.reshape((h*w,ch)).tolist())
# out_img = np.array(label).reshape((h,w))

out_img = np.array(Image.open(sys.argv[1]))
# print out_img

print out_img.getpalette()

# segfile = sys.argv[2]

# h, w = out_img.shape
# out_img = map(lambda x:index2color[x[0]],out_img.reshape((h*w,1)).tolist())
# out_img = np.array(out_img).reshape((h,w,3))
# cv2.imwrite(segfile,out_img)

# cv2.imwrite(segfile,out_img)

