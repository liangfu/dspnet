#!/usr/bin/env bash

echo '' > task.txt

range=`cat ImageSets/Segmentation/trainval.txt`

for ii in $range
do
    echo "echo $ii && python palette2grayscale.py SegmentationClass/$ii.png SegmentationClass/${ii}_index.png" >> task.txt
done

parallel -j2 < task.txt


