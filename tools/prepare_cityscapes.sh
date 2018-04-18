#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python $DIR/prepare_dataset.py --dataset cityscapes --set val --target $DIR/../data/val.lst --root $DIR/../data/cityscapes --shuffle=False
python $DIR/prepare_dataset.py --dataset cityscapes --set train --target $DIR/../data/train.lst --root $DIR/../data/cityscapes --shuffle=False
# python $DIR/prepare_dataset.py --dataset cityscapes --set test --target $DIR/../data/test.lst --root $DIR/../data/cityscapes --shuffle=False
# python $DIR/prepare_dataset.py --dataset cityscapes --set demoVideo --target $DIR/../data/demoVideo.lst --root $DIR/../data/cityscapes --shuffle=False
