#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os, sys
import cv2
import xml.etree.ElementTree as et
import math

DEBUG=True

def get_names(fname):
    names = []
    with open(sys.argv[1]) as fp:
        while 1:
            n = fp.readline().strip()
            if len(n)<1: break
            names.append(n)
    return names

def put_text(im, text, bbox):
    cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0,255,0), thickness=1)
    color_white = (255, 255, 255)
    fontFace = cv2.FONT_HERSHEY_PLAIN
    fontScale = .6
    thickness = 1
    textSize, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
    cv2.rectangle(im, (bbox[0], bbox[1]-textSize[1]), (bbox[0]+textSize[0], bbox[1]), color=(128,0,0), thickness=-1)
    cv2.putText(im, text, (bbox[0], bbox[1]),
                color=color_white, fontFace=fontFace, fontScale=fontScale, thickness=thickness)

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    import xml.dom.minidom
    rough_string = et.tostring(elem, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml()

def main():
    names = get_names(sys.argv[1]) # names file
    xmlname = sys.argv[2] # annotation file
    tree = et.parse(xmlname)
    root = tree.getroot()

    annotation = root.find("annotation")
    filename = root.find("filename").text
    img = cv2.imread(os.path.join("JPEGImages",filename))
    disparity = cv2.imread(os.path.join("Disparity",filename.replace("leftImg8bit.jpg","disparity.png")),-1).astype(np.float32)

    objects = root.findall("object")
    
    for obj in objects:
        name = obj.find("name").text
        if name in names:
            bndbox = obj.find("bndbox")
            xmin, xmax = int(bndbox.find("xmin").text), int(bndbox.find("xmax").text)
            ymin, ymax = int(bndbox.find("ymin").text), int(bndbox.find("ymax").text)
            xmin, ymin = max(0,xmin), max(0,ymin)
            if xmin==xmax:
                xmax=xmin+1
            roi = disparity[ymin:ymax,xmin:xmax]
            roi = np.sort(roi.reshape((1,-1)))
            # print name, roi.shape, xmin, xmax, ymin, ymax
            dist = 2200.*75./(roi[0,int(math.ceil(roi.shape[1]/2))]+1e-3)
            if dist>1000: dist = 200
            distance_tags = obj.findall("distance")
            for tag in distance_tags:
                obj.remove(tag)
            distance = et.SubElement(obj, "distance")
            distance.text = str(int(round(dist)))
            if DEBUG:
                text = '%s %.0fm' % (name, dist)
                put_text(img, text, [xmin,ymin,xmax,ymax])
            
    pretty_xml_as_string = et.tostring(root, 'utf-8')
    # if DEBUG:
    #     print pretty_xml_as_string
    with open(xmlname,"wt") as fp:
        fp.write(pretty_xml_as_string)
        
    if DEBUG:
        cv2.imshow(filename,img)
        cv2.waitKey()

if __name__=="__main__":
    main()

