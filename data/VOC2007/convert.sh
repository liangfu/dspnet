#!/usr/bin/env bash

range=`cat ImageSets/Segmentation/trainval.txt`

for ii in $range
do
    echo $ii && python palette2grayscale.py SegmentationClass/$ii.png SegmentationClass/${ii}_index.png 
done


