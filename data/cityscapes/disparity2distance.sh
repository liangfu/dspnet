#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "" > task.txt

files=`find ./Annotations/ | grep .xml`
# files=./Annotations/hamburg_000000_019892_leftImg8bit.xml
for f in $files
do
    echo "echo '$f' && python -u disparity2distance.py $DIR/../../dataset/names/cityscapes.txt $f" >> task.txt
done

parallel -j1 < task.txt
