#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export MXNET_EXAMPLE_SSD_DISABLE_PRE_INSTALLED=1
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
# export MXNET_ENGINE_TYPE=NaiveEngine

# dataset=VOC2012
# dataset=StreetScenes
dataset=Cityscapes
datashape=3,512,1024
epoch=300 # 545
# datashape=3,320,640
# epoch=865
classnames=dataset/names/cityscapes.txt
num_class=8
labelwidth=1202 #910
batch_size=1
end_epoch=2000
gpus=2
lr=0.0005

################## train resnet-50_det model ################
# python -u multi_train.py --pretrained=models/resnet-50 --network=resnet-50_det --class-names=$classnames --num-class=$num_class --data-shape=$datashape --label-width=$labelwidth --lr=$lr --batch-size=$batch_size --end-epoch=$end_epoch --gpus=$gpus
# python -u multi_train.py --pretrained=models/resnet-50 --network=resnet-50_det --class-names=$classnames --num-class=$num_class --data-shape=$datashape --label-width=$labelwidth --lr=$lr --batch-size=$batch_size --resume=$epoch --end-epoch=$end_epoch --gpus=$gpus

################## train resnet-50_seg model ################
# python -u multi_train.py --pretrained=models/resnet-50 --network=resnet-50_seg --class-names=$classnames --num-class=$num_class --data-shape=$datashape --label-width=$labelwidth --lr=$lr --batch-size=$batch_size --end-epoch=$end_epoch --gpus=$gpus
# python -u multi_train.py --pretrained=models/resnet-50 --network=resnet-50_seg --class-names=$classnames --num-class=$num_class --data-shape=$datashape --label-width=$labelwidth --lr=$lr --batch-size=$batch_size --resume=$epoch --end-epoch=$end_epoch --gpus=$gpus

################## train resnet-50_multi model ################
# python -u multi_train.py --pretrained=models/resnet-50 --network=resnet-50_multi --class-names=$classnames --num-class=$num_class --data-shape=$datashape --label-width=$labelwidth --lr=$lr --batch-size=$batch_size --end-epoch=$end_epoch --gpus=$gpus
# python -u multi_train.py --pretrained=models/resnet-50 --network=resnet-50_multi --class-names=$classnames --num-class=$num_class --data-shape=$datashape --label-width=$labelwidth --lr=$lr --batch-size=$batch_size --resume=$epoch --end-epoch=$end_epoch --gpus=$gpus

##### evaluation with demoVideo data
# python -u multi_train.py --pretrained=models/resnet-50 --network=resnet-50_multi --class-names=$classnames --num-class=$num_class --data-shape=$datashape --label-width=$labelwidth --lr=$lr --batch-size=$batch_size --resume=$epoch --end-epoch=$end_epoch --gpus=$gpus --val-path=$DIR/data/demoVideo.rec

################## demo with street image ################
python -u multi_demo.py --network=resnet-50_multi --class-names=$classnames --data-shape=$datashape --epoch=$epoch --thresh=.15 --images data/demo/aachen.jpg --nms=.3 --force=False
# python -u multi_demo.py --network=resnet-50_multi --class-names=$classnames --data-shape=$datashape --epoch=$epoch --thresh=.5 --images data/demo/stuttgart.mp4 --nms=.3 --force=False
# python -u multi_demo.py --network=resnet-50_multi --class-names=$classnames --data-shape=$datashape --epoch=$epoch --thresh=.25 --images 0 --nms=.3 --force=False

################## evalutation with validation set ################
# python -u multi_eval.py --network=resnet-50_multi --batch-size=1 --num-class $num_class --class-names=$classnames --data-shape=$datashape --epoch=$epoch --nms=.3 --overlap=.5 --force=True


