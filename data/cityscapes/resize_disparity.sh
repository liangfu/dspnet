#!/usr/bin/env bash

echo "" > task.txt

files=`find ./disparity_trainvaltest/ | grep .png`
for f in $files
do
    base=`basename $f`
    echo "echo $base && convert $f -interpolate nearest -filter point -resize 1024x512 Disparity/$base" >> task.txt
done

parallel -j4 < task.txt
