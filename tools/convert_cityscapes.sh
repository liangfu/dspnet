#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

############ CONVERT JSON FILES INTO PASCAL VOC COMPATIBLE XML FORMAT ############

echo "" > task.txt
files=`find $DIR/../data/cityscapes/gtFine_trainvaltest/ | grep .json`
for f in $files
do
    echo "echo $f && python $DIR/../dataset/cs_json2xml.py $f" >> task.txt
done
# parallel -j4 < task.txt

################ RENAME GENERATED XML FILES TO IMAGE BASE NAME ############

files=`find $DIR/../data/cityscapes/gtFine_trainvaltest/ | grep .xml`
for f in $files
do
    b=`basename $f`
    f2=$DIR/../data/cityscapes/Annotations/${b:0:${#b}-19}leftImg8bit.xml
    echo $f2 && cp $f $f2
done



