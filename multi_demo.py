import os
os.environ["MXNET_EXAMPLE_SSD_DISABLE_PRE_INSTALLED"]='1'

import argparse
import tools.find_mxnet
import mxnet as mx
import sys
from detect.multitask_detector import Detector
from symbol.multitask_symbol_factory import get_det_symbol, get_seg_symbol, get_multi_symbol

def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx, num_class,
                 nms_thresh=0.5, force_nms=True, nms_topk=400):
    """
    wrapper for initialize a detector

    Parameters:
    ----------
    net : str
        test network name
    prefix : str
        load model prefix
    epoch : int
        load model epoch
    data_shape : int
        resize image shape
    mean_pixels : tuple (float, float, float)
        mean pixel values (R, G, B)
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(?)
    num_class : int
        number of classes
    nms_thresh : float
        non-maximum suppression threshold
    force_nms : bool
        force suppress different categories
    """
    if net is not None:
        if net.endswith("fcn32s"):
            net = get_fcn32s_symbol(net.split("_")[0], data_shape, num_classes=num_class, nms_thresh=nms_thresh,
                force_nms=force_nms, nms_topk=nms_topk)
        elif net.endswith("fcn16s"):
            net = get_fcn16s_symbol(net.split("_")[0], data_shape, num_classes=num_class, nms_thresh=nms_thresh,
                force_nms=force_nms, nms_topk=nms_topk)
        elif net.endswith("fcn8s"):
            net = get_fcn8s_symbol(net.split("_")[0], data_shape, num_classes=num_class, nms_thresh=nms_thresh,
                force_nms=force_nms, nms_topk=nms_topk)

    ############### uncomment the following lines to visualize network ###########################
    # dot = mx.viz.plot_network(net, shape={'data':(1,3,512,1024)})
    # dot = mx.viz.plot_network(net, shape={'data':(1,3,320,640)})
    # dot.view()
        
    detector = Detector(net, prefix, epoch, data_shape, mean_pixels, ctx=ctx)
    return detector

def parse_args():
    parser = argparse.ArgumentParser(description='Single-shot detection network demo')
    parser.add_argument('--network', dest='network', type=str, default='resnet-18_fcn32s',
                        help='which network to use')
    parser.add_argument('--images', dest='images', type=str, default='./data/demo/dog.jpg',
                        help='run demo with images, use comma to seperate multiple images')
    parser.add_argument('--dir', dest='dir', nargs='?',
                        help='demo image directory, optional', type=str)
    parser.add_argument('--ext', dest='extension', help='image extension, optional',
                        type=str, nargs='?')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default=os.path.join(os.getcwd(), 'models', 'multitask_'),
                        type=str)
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to detect with')
    parser.add_argument('--data-shape', dest='data_shape', type=str, default="3,512,1024",
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.6,
                        help='object visualize score threshold, default 0.6')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--force', dest='force_nms', type=bool, default=True,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--timer', dest='show_timer', type=bool, default=True,
                        help='show detection time')
    parser.add_argument('--deploy', dest='deploy_net', action='store_true', default=False,
                        help='Load network from json file, rather than from symbol')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='aeroplane, bicycle, bird, boat, bottle, bus, \
                        car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                        person, pottedplant, sheep, sofa, train, tvmonitor',
                        help='string of comma separated names, or text filename')
    args = parser.parse_args()
    return args

def parse_class_names(class_names):
    """ parse # classes and class_names if applicable """
    if len(class_names) > 0:
        if os.path.isfile(class_names):
            # try to open it to read class names
            with open(class_names, 'r') as f:
                class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in class_names.split(',')]
        for name in class_names:
            assert len(name) > 0
    else:
        raise RuntimeError("No valid class_name provided...")
    return class_names

if __name__ == '__main__':
    args = parse_args()
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)

    # parse image list
    # image_list = [i.strip() for i in args.images.split(',')]
    # assert len(image_list) > 0, "No valid image specified to detect"
    imgname = args.images

    network = None if args.deploy_net else args.network
    class_names = parse_class_names(args.class_names)
    data_shape = None
    if isinstance(args.data_shape, int):
        data_shape = 3,args.data_shape,args.data_shape
    else:
        data_shape = map(lambda x:int(x),args.data_shape.split(","))
        assert len(data_shape) == 3 and data_shape[0] == 3
        
    if args.prefix.endswith('_'):
        prefix = args.prefix + args.network + '_' + str(data_shape[1])
    else:
        prefix = args.prefix

    # print(network, prefix, args.epoch, data_shape,
    #       (args.mean_r, args.mean_g, args.mean_b),
    #       ctx, len(class_names), args.nms_thresh, args.force_nms)
    detector = get_detector(network, prefix, args.epoch, data_shape,
                            (args.mean_r, args.mean_g, args.mean_b),
                            ctx, len(class_names), args.nms_thresh, args.force_nms)
    # run detection
    detector.detect_and_visualize(imgname, args.dir, args.extension,
                                  class_names, args.thresh, args.show_timer)
