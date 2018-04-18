#!/usr/bin/env python

from __future__ import print_function
import os
os.environ["MXNET_EXAMPLE_SSD_DISABLE_PRE_INSTALLED"]="1"
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"]="0"

from dataset.iterator import MultiTaskRecordIter
from config.config import cfg
from evaluate.eval_metric import MApMetric, IoUMetric
import logging
from symbol.multitask_symbol_factory import get_det_symbol, get_seg_symbol, get_multi_symbol
from train.metric import MultiBoxMetric, CustomAccuracyMetric, DistanceAccuracyMetric
from mxnet import metric
import numpy as np
from utils import put_text
import time
import argparse
import tools.find_mxnet
import mxnet as mx
import os
import sys
import cv2

DEBUG=False
outimgiter=0

affine_matrix = mx.nd.array([[1, 0, 0],[0, 1, 0]],ctx=mx.gpu(0))
affine_matrix = mx.nd.reshape(affine_matrix, shape=(1, 6))
GRID = mx.nd.GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(1024,2048))
def prob_upsampling(seg_prob, target_shape):
    seg_prob = mx.nd.BilinearSampler(mx.nd.expand_dims(seg_prob,axis=0), GRID)
    seg_resized = np.squeeze(mx.nd.argmax(seg_prob,axis=1).asnumpy()).astype(np.uint8)
    return seg_resized

def display_results(out_img,label_img,img,det,gt_boxes,class_names):
    # from utils import getpallete
    # palette = getpallete(256)
    from dataset.cs_labels import labels
    lut = np.zeros((256,3))
    for l in labels:
        if l.trainId<255 and l.trainId>=0:
            lut[l.trainId,:]=list(l.color)
    palette = lut
    
    from detect.nms import nms
    import cv2
    if DEBUG:
        print({"out_img":out_img.shape,"label_img":label_img.shape,"img":img.shape})
    lut_b = np.array(palette).astype(np.uint8).reshape((256,3))[:,0]
    lut_g = np.array(palette).astype(np.uint8).reshape((256,3))[:,1]
    lut_r = np.array(palette).astype(np.uint8).reshape((256,3))[:,2]
    # print np.vstack((lut_r[:10],lut_g[:10],lut_b[:10]))
    # out_img = np.squeeze(self.executor.outputs[0].asnumpy().argmax(axis=1).astype(np.uint8))
    out_img_r = cv2.LUT(out_img,lut_r)
    out_img_g = cv2.LUT(out_img,lut_g)
    out_img_b = cv2.LUT(out_img,lut_b)
    out_img = cv2.merge((out_img_r,out_img_g,out_img_b))
    # label_img = data[label_name].astype(np.uint8)
    label_img = np.swapaxes(label_img, 1, 2)
    label_img = np.swapaxes(label_img, 0, 2).astype(np.uint8)
    label_img_r = cv2.LUT(label_img,lut_r)
    label_img_g = cv2.LUT(label_img,lut_g)
    label_img_b = cv2.LUT(label_img,lut_b)
    label_img = cv2.merge((label_img_r,label_img_g,label_img_b))
    # img = np.squeeze(data[data_name])
    img = (img + np.array([123.68, 116.779, 103.939]).reshape((3,1,1))).astype(np.uint8)
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 0, 2).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # detection result
    det_img = img.copy()
    dets = det[np.where(det[:,0]>=0),:].reshape((-1,7))
    if DEBUG:
        print(dets[:2,:])
    # idx = nms(np.hstack((dets[:,2:6],dets[:,1:2])),.9)
    # dets = dets[idx,:]
    h, w, ch = img.shape
    for idx in range(dets.shape[0]):
        # if dets[idx,1]<.15:
        #     continue
        bbox = [int(round(dets[idx,2]*w)),int(round(dets[idx,3]*h)), \
                int(round(dets[idx,4]*w)),int(round(dets[idx,5]*h))]
        cv2.rectangle(det_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0,0,128), thickness=1)
        text = "%s:%.0fm" % (class_names[int(dets[idx,0])],dets[idx,6]*255.,)
        put_text(det_img, text, bbox, fontScale=.8)
    for box in gt_boxes.tolist():
        bbox = [int(round(box[1]*w)),int(round(box[2]*h)), \
                int(round(box[3]*w)),int(round(box[4]*h))]
        if (bbox[2]-bbox[0])*(bbox[3]-bbox[1])<100:
            continue
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0,0,128), thickness=1)
        text = "%s:%.0fm" % (class_names[int(box[0])],box[5]*255.,)
        put_text(img, text, bbox, fontScale=.8)
    if DEBUG:
        print({"img.shape":img.shape,"label_img.shape":label_img.shape,"out_img.shape":out_img.shape})
    if img.shape[0]!=out_img.shape[0]:
        out_img = cv2.resize(out_img,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
        label_img = cv2.resize(label_img,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
    displayimg = np.vstack((np.hstack((img,label_img)), np.hstack((det_img,out_img))))
    displayimg_resized = cv2.resize(displayimg, (int(w*2*.8),int(h*2*.8)))
    cv2.imshow('out_img',displayimg_resized)
    # cv2.imwrite('tmp/out_img_%03d.png'%(outimgiter,),displayimg);
    return displayimg

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a network')
    parser.add_argument('--rec-path', dest='rec_path', help='which record file to use',
                        default=os.path.join(os.getcwd(), 'data', 'val.rec'), type=str)
    parser.add_argument('--list-path', dest='list_path', help='which list file to use',
                        default="", type=str)
    parser.add_argument('--network', dest='network', type=str, default='resnet-18_fcn16s',
                        help='which network to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                        help='evaluation batch size')
    parser.add_argument('--num-class', dest='num_class', type=int, default=20,
                        help='number of classes')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='aeroplane, bicycle, bird, boat, bottle, bus, \
                        car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                        person, pottedplant, sheep, sofa, train, tvmonitor',
                        help='string of comma separated names, or text filename')
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='load model prefix',
                        default=os.path.join(os.getcwd(), 'models', 'multitask_'), type=str)
    parser.add_argument('--gpus', dest='gpu_id', help='GPU devices to evaluate with',
                        default='0', type=str)
    parser.add_argument('--cpu', dest='cpu', help='use cpu to evaluate, this can be slow',
                        action='store_true')
    parser.add_argument('--data-shape', dest='data_shape', type=str, default="3,512,1024",
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.45,
                        help='non-maximum suppression threshold')
    parser.add_argument('--overlap', dest='overlap_thresh', type=float, default=0.5,
                        help='evaluation overlap threshold')
    parser.add_argument('--force', dest='force_nms', type=bool, default=False,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--use-difficult', dest='use_difficult', type=bool, default=True,
                        help='use difficult ground-truths in evaluation')
    parser.add_argument('--voc07', dest='use_voc07_metric', type=bool, default=False,
                        help='use PASCAL VOC 07 metric')
    parser.add_argument('--deploy', dest='deploy_net', help='Load network from model',
                        action='store_true', default=False)
    args = parser.parse_args()
    return args

def evaluate(netname, path_imgrec, num_classes, num_seg_classes, mean_pixels, data_shape,
                 model_prefix, epoch, ctx=mx.cpu(), batch_size=1,
                 path_imglist="", nms_thresh=0.45, force_nms=False,
                 ovp_thresh=0.5, use_difficult=False, class_names=None, seg_class_names=None,
                 voc07_metric=False):
    """
    evalute network given validation record file

    Parameters:
    ----------
    net : str or None
        Network name or use None to load from json without modifying
    path_imgrec : str
        path to the record validation file
    path_imglist : str
        path to the list file to replace labels in record file, optional
    num_classes : int
        number of classes, not including background
    mean_pixels : tuple
        (mean_r, mean_g, mean_b)
    data_shape : tuple or int
        (3, height, width) or height/width
    model_prefix : str
        model prefix of saved checkpoint
    epoch : int
        load model epoch
    ctx : mx.ctx
        mx.gpu() or mx.cpu()
    batch_size : int
        validation batch size
    nms_thresh : float
        non-maximum suppression threshold
    force_nms : boolean
        whether suppress different class objects
    ovp_thresh : float
        AP overlap threshold for true/false postives
    use_difficult : boolean
        whether to use difficult objects in evaluation if applicable
    class_names : comma separated str
        class names in string, must correspond to num_classes if set
    voc07_metric : boolean
        whether to use 11-point evluation as in VOC07 competition
    """
    global outimgiter
    
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # args
    if isinstance(data_shape, int):
        data_shape = (3, data_shape, data_shape)
    else:
        data_shape = map(int,data_shape.split(","))
    assert len(data_shape) == 3 and data_shape[0] == 3
    model_prefix += '_' + str(data_shape[1])

    # iterator
    eval_iter = MultiTaskRecordIter(path_imgrec, batch_size, data_shape, 
                                    path_imglist=path_imglist, enable_aug=False, **cfg.valid)
    # model params
    load_net, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
    # network
    if netname is None:
        net = load_net
    elif netname.endswith("det"):
        net = get_det_symbol(netname.split("_")[0], data_shape[1], num_classes=num_classes,
            nms_thresh=nms_thresh, force_suppress=force_nms)
    elif netname.endswith("seg"):
        net = get_seg_symbol(netname.split("_")[0], data_shape[1], num_classes=num_classes,
            nms_thresh=nms_thresh, force_suppress=force_nms)
    elif netname.endswith("multi"):
        net = get_multi_symbol(netname.split("_")[0], data_shape[1], num_classes=num_classes,
            nms_thresh=nms_thresh, force_suppress=force_nms)
    else:
        raise NotImplementedError("")
    
    if not 'label_det' in net.list_arguments():
        label_det = mx.sym.Variable(name='label_det')
        net = mx.sym.Group([net, label_det])
    if not 'seg_out_label' in net.list_arguments():
        seg_out_label = mx.sym.Variable(name='seg_out_label')
        net = mx.sym.Group([net, seg_out_label])

    # init module
    # mod = mx.mod.Module(net, label_names=('label_det','seg_out_label',), logger=logger, context=ctx,
    #     fixed_param_names=net.list_arguments())
    # mod.bind(data_shapes=eval_iter.provide_data, label_shapes=eval_iter.provide_label)
    # mod.set_params(args, auxs, allow_missing=False, force_init=True)
    # metric = MApMetric(ovp_thresh, use_difficult, class_names)
    # results = mod.score(eval_iter, metric, num_batch=None)
    # for k, v in results:
    #     print("{}: {}".format(k, v))

    ctx = ctx[0]
    eval_metric = CustomAccuracyMetric()
    multibox_metric = MultiBoxMetric()
    depth_metric = DistanceAccuracyMetric(class_names=class_names)
    det_metric = MApMetric(ovp_thresh, use_difficult, class_names)
    seg_metric = IoUMetric(class_names=seg_class_names, axis=1)
    eval_metrics = metric.CompositeEvalMetric()
    eval_metrics.add(multibox_metric)
    eval_metrics.add(eval_metric)
    arg_params = {key: val.as_in_context(ctx) for key, val in args.items()}
    aux_params = {key: val.as_in_context(ctx) for key, val in auxs.items()}
    data_name = eval_iter.provide_data[0][0]
    label_name_det = eval_iter.provide_label[0][0]
    label_name_seg = eval_iter.provide_label[1][0]
    symbol = load_net
    
    # evaluation
    logger.info(" in eval process...")
    logger.info(str({"ovp_thresh":ovp_thresh,"nms_thresh":nms_thresh,"batch_size":batch_size,
                     "force_nms":force_nms,}))
    nbatch = 0
    eval_iter.reset()
    eval_metrics.reset()
    det_metric.reset()
    total_time = 0
    
    for data, fnames in eval_iter:
        nbatch += 1
        label_shape_det = data.label[0].shape
        label_shape_seg = data.label[1].shape
        arg_params[data_name] = mx.nd.array(data.data[0], ctx)
        arg_params[label_name_det] = mx.nd.array(data.label[0], ctx)
        arg_params[label_name_seg] = mx.nd.array(data.label[1], ctx)
        executor = symbol.bind(ctx, arg_params, aux_states=aux_params)

        output_names = symbol.list_outputs()
        output_dict = dict(zip(output_names, executor.outputs))

        cpu_output_array = mx.nd.zeros(output_dict["seg_out_output"].shape)

        ############## monitor status
        def stat_helper(name, array):
            """wrapper for executor callback"""
            import ctypes
            from mxnet.ndarray import NDArray
            from mxnet.base import NDArrayHandle, py_str
            array = ctypes.cast(array, NDArrayHandle)
            if 1:
                array = NDArray(array, writable=False).asnumpy()
                print (name, array.shape, np.mean(array), np.std(array),
                       ('%.1fms' % (float(time.time()-stat_helper.start_time)*1000)))
            else:
                array = NDArray(array, writable=False)
                array.wait_to_read()
                elapsed = float(time.time()-stat_helper.start_time)*1000.
                if elapsed>5:
                    print (name, array.shape, ('%.1fms' % (elapsed,)))
            stat_helper.start_time=time.time()
        stat_helper.start_time=float(time.time())
        # executor.set_monitor_callback(stat_helper)

        ############## forward
        tic = time.time()
        executor.forward(is_train=True)
        output_dict["seg_out_output"].copyto(cpu_output_array)
        pred_shape = output_dict["seg_out_output"].shape
        label = mx.nd.array(data.label[1].reshape((label_shape_seg[0], label_shape_seg[1]*label_shape_seg[2])))
        output_dict["seg_out_output"].wait_to_read()

        toc = time.time()
        
        seg_out_output = output_dict["seg_out_output"].asnumpy()

        pred_seg_shape = output_dict["seg_out_output"].shape
        label_det = mx.nd.array(data.label[0].reshape((label_shape_det[0],
            label_shape_det[1]*label_shape_det[2])))
        label_seg = mx.nd.array(data.label[1].reshape((label_shape_seg[0],
            label_shape_seg[1]*label_shape_seg[2])),ctx=ctx)
        pred_seg = mx.nd.array(output_dict["seg_out_output"].reshape((pred_seg_shape[0],
            pred_seg_shape[1], pred_seg_shape[2]*pred_seg_shape[3])),ctx=ctx)
        #### remove invalid boxes 
        out_det = output_dict["det_out_output"].asnumpy()
        indices = np.where(out_det[:,:,0]>=0) # labeled as negative
        out_det = np.expand_dims(out_det[indices[0],indices[1],:],axis=0)
        indices = np.where(out_det[:,:,1]>.1) # higher confidence
        out_det = np.expand_dims(out_det[indices[0],indices[1],:],axis=0)
        # indices = np.where(out_det[:,:,6]<=(100/255.)) # too far away
        # out_det = np.expand_dims(out_det[indices[0],indices[1],:],axis=0)
        pred_det = mx.nd.array(out_det)
        #### remove labels too faraway
        # label_det = label_det.asnumpy().reshape((200,6))
        # indices = np.where(label_det[:,5]<=(100./255.))
        # label_det = np.expand_dims(label_det[indices[0],:],axis=0)
        # label_det = mx.nd.array(label_det)

        ################# display results ####################
        out_img = output_dict["seg_out_output"]
        out_img = mx.nd.split(out_img, axis=0, num_outputs=out_img.shape[0])
        for imgidx in range(batch_size):
            seg_prob = out_img[imgidx]
            res_img = np.squeeze(seg_prob.asnumpy().argmax(axis=0).astype(np.uint8))
            label_img = data.label[1].asnumpy()[imgidx,:,:].astype(np.uint8)
            img = np.squeeze(data.data[0].asnumpy()[imgidx,:,:,:])
            det = out_det[imgidx,:,:]
            gt = label_det.asnumpy()[imgidx,:].reshape((-1,6))
            # save to results folder for evalutation
            res_fname = fnames[imgidx].replace("SegmentationClass","results")
            lut = np.zeros(256)
            lut[:19]=np.array([7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33])
            seg_resized = prob_upsampling(seg_prob, target_shape=(1024,2048))
            seg_resized2 = cv2.LUT(seg_resized,lut)
            # seg = cv2.LUT(res_img,lut)
            # cv2.imshow("seg",seg.astype(np.uint8))
            cv2.imwrite(res_fname, seg_resized2)
            # display result
            print(fnames[imgidx],np.average(img))
            display_img = display_results(res_img,np.expand_dims(label_img,axis=0),img, det, gt, class_names)
            res_fname = fnames[imgidx].replace("SegmentationClass","output").replace("labelTrainIds","output")
            cv2.imwrite(res_fname, display_img)
            [exit(0) if (cv2.waitKey()&0xff)==27 else None]
        outimgiter += 1
        ################# display results ####################

        eval_metrics.get_metric(0).update(None,
                               [output_dict["cls_prob_output"], output_dict["loc_loss_output"],
                                output_dict["cls_label_output"]])
        eval_metrics.get_metric(1).update([label_seg], [pred_seg])
        det_metric.update([mx.nd.slice_axis(data.label[0],axis=2,begin=0,end=5)], \
                                 [mx.nd.slice_axis(pred_det,axis=2,begin=0,end=6)])
        seg_metric.update([label_seg], [pred_seg])
        disparities = []
        for imgidx in range(batch_size):
            dispname = fnames[imgidx].replace("SegmentationClass","Disparity").replace("gtFine_labelTrainIds","disparity")
            print(dispname)
            disparities.append(cv2.imread(dispname,-1))
        depth_metric.update(mx.nd.array(disparities),[pred_det])
        
        det_names, det_values = det_metric.get()
        seg_names, seg_values = seg_metric.get()
        depth_names, depth_values = depth_metric.get()
        total_time += toc-tic
        print("\r %d/%d %.1f%% speed=%.1fms %s=%.1f %s=%.1f %s=%.1f" %
              (nbatch*eval_iter.batch_size,eval_iter.num_samples,
            float(nbatch*eval_iter.batch_size)*100./float(eval_iter.num_samples),
            total_time*1000./nbatch,
            det_names[-1],det_values[-1]*100.,
            seg_names[-1],seg_values[-1]*100.,
            depth_names[-1],depth_values[-1]*100.,),end='\r')

        # if nbatch>50: break ## debugging
        
    names, values = eval_metrics.get()
    for name, value in zip(names,values):
        logger.info(' epoch[%d] Validation-%s=%f', epoch, name, value)
    logger.info('----------------------------------------------')
    names, values = det_metric.get()
    for name, value in zip(names,values):
        logger.info(' epoch[%d] Validation-%s=%f', epoch, name, value)
    logger.info('----------------------------------------------')
    logger.info(' & '.join(names))
    logger.info(' & '.join(map(lambda v:'%.1f'%(v*100.,),values)))
    logger.info('----------------------------------------------')
    names, values = depth_metric.get()
    for name, value in zip(names,values):
        logger.info(' epoch[%d] Validation-%s=%f', epoch, name, value)
    logger.info('----------------------------------------------')
    logger.info(' & '.join(names))
    logger.info(' & '.join(map(lambda v:'%.1f'%(v*100.,),values)))
    logger.info('----------------------------------------------')
    names, values = seg_metric.get()
    for name, value in zip(names,values):
        logger.info(' epoch[%d] Validation-%s=%f', epoch, name, value)
    logger.info('----------------------------------------------')
    logger.info(' & '.join(names))
    logger.info(' & '.join(map(lambda v:'%.1f'%(v*100.,),values)))
    

if __name__ == '__main__':
    args = parse_args()
    # choose ctx
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpu_id.split(',')]
    # parse # classes and class_names if applicable
    num_class = args.num_class
    if len(args.class_names) > 0:
        if os.path.isfile(args.class_names):
                # try to open it to read class names
                with open(args.class_names, 'r') as f:
                    class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in args.class_names.split(',')]
        assert len(class_names) == num_class
        for name in class_names:
            assert len(name) > 0
    else:
        class_names = None

    # parse segmentation classes
    with open("dataset/names/cityscapes_seg.txt", 'r') as f:
        seg_class_names = [l.strip() for l in f.readlines()]
    num_seg_class = len(seg_class_names)
        
    network = None if args.deploy_net else args.network
    if args.prefix.endswith('_'):
        prefix = args.prefix + args.network
    else:
        prefix = args.prefix
    print(args)
    evaluate(network, args.rec_path, num_class, num_seg_class,
             (args.mean_r, args.mean_g, args.mean_b), args.data_shape,
             prefix, args.epoch, ctx, batch_size=args.batch_size,
             path_imglist=args.list_path, nms_thresh=args.nms_thresh,
             force_nms=args.force_nms, ovp_thresh=args.overlap_thresh,
             use_difficult=args.use_difficult, class_names=class_names,
             seg_class_names=seg_class_names, voc07_metric=args.use_voc07_metric)
