# pylint: skip-file
from __future__ import print_function
import numpy as np
import mxnet as mx
import time
import logging
from collections import namedtuple
from mxnet import optimizer as opt
from mxnet.optimizer import get_updater
from mxnet import metric
from pprint import pprint
from train.metric import MultiBoxMetric, CustomAccuracyMetric, DistanceAccuracyMetric
from utils import put_text
from dataset.cs_labels import labels as cs_labels
from evaluate.eval_metric import MApMetric
import cv2
from detect.nms import nms
import math, os, sys

outimgiter = 0
DEBUG = False
TIMING = False
short_class_name = {"traffic light":"t-light","traffic sign":"t-sign","person":"person",\
                  "rider":"rider","car":"car","truck":"truck","bus":"bus","train":"train",\
                    "motorcycle":"mbike","bicycle":"bike","vegetation":"tree"}

affine_matrix = mx.nd.array([[1, 0, 0],[0, 1, 0]],ctx=mx.gpu(0))
affine_matrix = mx.nd.reshape(affine_matrix, shape=(1, 6))
GRID = mx.nd.GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(1024,2048))
def prob_upsampling(seg_prob, target_shape):
    seg_prob = mx.nd.BilinearSampler(mx.nd.expand_dims(seg_prob,axis=0), GRID)
    seg_resized = np.squeeze(mx.nd.argmax(seg_prob,axis=1).asnumpy()).astype(np.uint8)
    return seg_resized

def get_seg_labels(shape):
    annotation = np.zeros(shape,np.uint8)
    from dataset.cs_labels import labels
    from palette import get_palette
    # palette = get_palette(256)
    from dataset.cs_labels import labels as cs_labels
    lut = np.zeros((256,3))
    labels = []
    for l in cs_labels:
        if l.trainId<255 and l.trainId>=0:
            labels.append((l.trainId,l.name,l.color))
    palette = lut.flatten()
    colors = np.array(palette).reshape((-1,3))
    padding, blocksize, notes = 100, 15, 10
    for idx,name,label in labels:
        color = label
        color = (color[2],color[1],color[0])
        if idx<notes:
            anchor = (idx*padding,0)
        elif idx<notes*2:
            anchor = ((idx-notes)*padding,blocksize)
        cv2.rectangle(annotation, anchor, (anchor[0]+blocksize, anchor[1]+blocksize), color=color, thickness=-1)
        fontFace = cv2.FONT_HERSHEY_PLAIN
        fontScale = .8
        # name = short_class_name[name] if name in short_class_name.keys() else name
        cv2.putText(annotation, name, (anchor[0]+blocksize+1, anchor[1]+10), color=(255,255,255), \
                    fontFace=fontFace, fontScale=fontScale)
    return annotation

def display_results(out_img,label_img,img,class_names):
    # from utils import getpallete
    # palette = getpallete(256)
    from dataset.cs_labels import labels
    lut = np.zeros((256,3))
    for l in labels:
        if l.trainId<255 and l.trainId>=0:
            lut[l.trainId,:]=list(l.color)
    palette = lut
    det2seg = {0:6,1:7,2:11,3:12,4:13,5:14,6:15,7:16,8:17,9:18,}

    if DEBUG:
        print({"out_img":out_img.shape,"label_img":label_img.shape,"img":img.shape})
    lut_reshaped = np.array(palette).astype(np.uint8).reshape((256,3))
    lut_b = lut_reshaped[:,0]
    lut_g = lut_reshaped[:,1]
    lut_r = lut_reshaped[:,2]
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
    # det_img = img.copy()
    # dets = det[np.where(det[:,0]>=0),:].reshape((-1,7))
    # if DEBUG:
    #     print(dets[:2,:])
    # idx = nms(np.hstack((dets[:,2:6],dets[:,1:2])),.85)
    # dets = dets[idx,:]
    # idx = np.argsort(dets[:,6],axis=0)[::-1] ## draw nearest first !!
    # dets = dets[idx,:]
    h, w, ch = img.shape
    fontScale = .8*(h/float(320))
    thickness = 2 if h>320 else 1
    # for idx in range(dets.shape[0]):
    #     # if dets[idx,1]<.15:
    #     #     continue
    #     # bbox = [int(round(dets[idx,2]*w)),int(round(dets[idx,3]*h)), \
    #     #         int(round(dets[idx,4]*w)),int(round(dets[idx,5]*h))]
    #     # color = palette[det2seg[int(dets[idx,0])],:]
    #     # cv2.rectangle(det_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(color[2],color[1],color[0]), thickness=thickness)
    #     # clsname = class_names[int(dets[idx,0])]
    #     # clsname_short = short_class_name[clsname]
    #     # text = "%s:%.0fm" % (clsname_short,dets[idx,6]*255.,)
    #     # text = "%.0fm" % (dets[idx,6]*255.,)
    #     # put_text(det_img, text, bbox, fontScale=fontScale)
    # for box in gt_boxes.tolist():
    #     bbox = [int(round(box[1]*w)),int(round(box[2]*h)), \
    #             int(round(box[3]*w)),int(round(box[4]*h))]
    #     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0,0,128), thickness=thickness)
    #     clsname = class_names[int(box[0])]
    #     clsname_short = short_class_name[clsname]
    #     text = "%s:%.0fm" % (clsname_short,box[5]*255.,)
    #     put_text(img, text, bbox, fontScale=fontScale)
    if DEBUG:
        print("img.shape,label_img.shape,out_img.shape", \
              img.shape,label_img.shape,out_img.shape)
    if img.shape[0]!=out_img.shape[0]:
        out_img = cv2.resize(out_img,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
        label_img = cv2.resize(label_img,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
    if 1: # for training data with labels
        displayimg = np.vstack((label_img,out_img))
    else: # for evaluation_only is ture
        displayimg = np.vstack((det_img,out_img))
        seg_labels = get_seg_labels((30, displayimg.shape[1],3))
        displayimg = np.vstack((displayimg,seg_labels))
    if False: #displayimg.shape[0]>1000:
        hh, ww, ch = displayimg.shape
        displayimg_resized = cv2.resize(displayimg, (int(ww*.8),int(hh*.8)))
    else:
        displayimg_resized = displayimg
    cv2.imshow('out_img',displayimg_resized);
    # [exit(0) if (cv2.waitKey()&0xff)==27 else None]
    # cv2.imwrite('tmp/out_img_%03d.png'%(outimgiter,),displayimg);
    return displayimg


# Parameter to pass to batch_end_callback
BatchEndParam = namedtuple('BatchEndParams', ['epoch', 'nbatch', 'eval_metric'])

class SegTaskSolver(object):
    def __init__(self, symbol, ctx=None,
                 begin_epoch=0, num_epoch=None,
                 arg_params=None, aux_params=None,
                 valid_metric=MApMetric(),
                 class_names=[],
                 optimizer='sgd', **kwargs):
        self.symbol = symbol
        if ctx is None:
            ctx = mx.cpu(0)
        self.ctx = ctx
        self.begin_epoch = begin_epoch
        self.num_epoch = num_epoch
        self.arg_params = arg_params
        self.aux_params = aux_params
        self.valid_metric = valid_metric
        self.class_names = class_names
        self.optimizer = optimizer
        self.evaluation_only = False
        self.kwargs = kwargs.copy()

    def fit(self, train_data, eval_data=None,
            eval_metric='acc',
            grad_req='write',
            epoch_end_callback=None,
            batch_end_callback=None,
            kvstore='local',
            logger=None):
        global outimgiter
        if logger is None:
            logger = logging
        logging.info('Start training with %s', str(self.ctx))
        logging.info(str(self.kwargs))
        batch_size = train_data.provide_data[0][1][0]
        arg_shapes, out_shapes, aux_shapes = self.symbol.infer_shape( data=tuple(train_data.provide_data[0][1]))
        arg_names = self.symbol.list_arguments()
        out_names = self.symbol.list_outputs()
        aux_names = self.symbol.list_auxiliary_states()

        # pprint([(n,s) for n,s in zip(arg_names,arg_shapes)])
        # pprint([(n,s) for n,s in zip(out_names,out_shapes)])
        # pprint([(n,s) for n,s in zip(aux_names,aux_shapes)])
        
        if grad_req != 'null':
            self.grad_params = {}
            for name, shape in zip(arg_names, arg_shapes):
                if not (name.endswith('data') or name.endswith('label')):
                    self.grad_params[name] = mx.nd.zeros(shape, self.ctx)
        else:
            self.grad_params = None
        self.aux_params = {k : mx.nd.zeros(s, self.ctx) for k, s in zip(aux_names, aux_shapes)}
        data_name = train_data.provide_data[0][0]
        label_name_det = train_data.provide_label[0][0]
        label_name_seg = train_data.provide_label[1][0]
        input_names = [data_name, label_name_det, label_name_seg]

        print(train_data.provide_label)
        print(os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"])

        self.optimizer = opt.create(self.optimizer, rescale_grad=(1.0/train_data.batch_size), **(self.kwargs))
        self.updater = get_updater(self.optimizer)
        eval_metric = CustomAccuracyMetric() # metric.create(eval_metric)
        multibox_metric = MultiBoxMetric()

        eval_metrics = metric.CompositeEvalMetric()
        # eval_metrics.add(multibox_metric)
        eval_metrics.add(eval_metric)
        
        # begin training
        for epoch in range(self.begin_epoch, self.num_epoch):
            nbatch = 0
            train_data.reset()
            eval_metrics.reset()
            logger.info('learning rate: '+str(self.optimizer.learning_rate))
            for data,_ in train_data:
                if self.evaluation_only:
                    break
                nbatch += 1
                label_shape_det = data.label[0].shape
                label_shape_seg = data.label[1].shape
                self.arg_params[data_name] = mx.nd.array(data.data[0], self.ctx)
                self.arg_params[label_name_det] = mx.nd.array(data.label[0], self.ctx)
                self.arg_params[label_name_seg] = mx.nd.array(data.label[1], self.ctx)
                output_names = self.symbol.list_outputs()

                ###################### analyze shapes ####################
                # pprint([(k,v.shape) for k,v in self.arg_params.items()])
                
                self.executor = self.symbol.bind(self.ctx, self.arg_params,
                    args_grad=self.grad_params, grad_req=grad_req, aux_states=self.aux_params)
                assert len(self.symbol.list_arguments()) == len(self.executor.grad_arrays)
                update_dict = {name: nd for name, nd in zip(self.symbol.list_arguments(), \
                    self.executor.grad_arrays) if nd is not None}
                output_dict = {}
                output_buff = {}
                for key, arr in zip(self.symbol.list_outputs(), self.executor.outputs):
                    output_dict[key] = arr
                    output_buff[key] = mx.nd.empty(arr.shape, ctx=mx.cpu())
                    # output_buff[key] = mx.nd.empty(arr.shape, ctx=self.ctx)

                def stat_helper(name, array):
                    """wrapper for executor callback"""
                    import ctypes
                    from mxnet.ndarray import NDArray
                    from mxnet.base import NDArrayHandle, py_str
                    array = ctypes.cast(array, NDArrayHandle)
                    if 0:
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
                # self.executor.set_monitor_callback(stat_helper)

                tic = time.time()
                    
                self.executor.forward(is_train=True)
                for key in output_dict:
                    output_dict[key].copyto(output_buff[key])

                # exit(0) # for debugging forward pass only
                    
                self.executor.backward()
                for key, arr in update_dict.items():
                    if key != "bigscore_weight":
                        self.updater(key, arr, self.arg_params[key])

                for output in self.executor.outputs:
                    output.wait_to_read()
                if TIMING:
                    print("%.0fms" % ((time.time()-tic)*1000.,))
                        
                output_dict = dict(zip(output_names, self.executor.outputs))
                # pred_det_shape = output_dict["det_out_output"].shape
                pred_seg_shape = output_dict["seg_out_output"].shape
                # label_det = mx.nd.array(data.label[0].reshape((label_shape_det[0],
                #                                                label_shape_det[1]*label_shape_det[2])))
                label_seg = mx.nd.array(data.label[1].reshape((label_shape_seg[0],
                                                               label_shape_seg[1]*label_shape_seg[2])))
                # pred_det = mx.nd.array(output_buff["det_out_output"].reshape((pred_det_shape[0],
                #     pred_det_shape[1], pred_det_shape[2])))
                pred_seg = mx.nd.array(output_buff["seg_out_output"].reshape((pred_seg_shape[0],
                    pred_seg_shape[1], pred_seg_shape[2]*pred_seg_shape[3])))
                if DEBUG:
                    print(data.label[0].asnumpy()[0,:2,:])

                if TIMING:
                    print("%.0fms" % ((time.time()-tic)*1000.,))
                    
                # eval_metrics.get_metric(0).update([mx.nd.zeros(output_buff["cls_prob_output"].shape),
                #                         mx.nd.zeros(output_buff["loc_loss_output"].shape),label_det],
                #                        [output_buff["cls_prob_output"], output_buff["loc_loss_output"],
                #                         output_buff["cls_label_output"]])
                eval_metrics.get_metric(0).update([label_seg.as_in_context(self.ctx)], [pred_seg.as_in_context(self.ctx)])

                self.executor.outputs[0].wait_to_read()

                ##################### display results ##############################
                # out_img = output_dict["seg_out_output"].asnumpy()
                # out_det = output_dict["det_out_output"].asnumpy()
                # for imgidx in range(out_img.shape[0]):
                #     res_img = np.squeeze(out_img[imgidx,:,:].argmax(axis=0).astype(np.uint8))
                #     label_img = data.label[1].asnumpy()[imgidx,:,:].astype(np.uint8)
                #     img = np.squeeze(data.data[0].asnumpy()[imgidx,:,:,:])
                #     det = out_det[imgidx,:,:]
                #     gt = label_det.asnumpy()[imgidx,:].reshape((-1,6))
                #     display_results(res_img,np.expand_dims(label_img,axis=0),img, self.class_names)
                #     [exit(0) if (cv2.waitKey()&0xff)==27 else None]
                # outimgiter += 1

                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, eval_metric=eval_metrics)
                batch_end_callback(batch_end_params)

                if TIMING:
                    print("%.0fms" % ((time.time()-tic)*1000.,))
                    
                # exit(0) # for debugging only
                
            ##### save snapshot
            if (not self.evaluation_only) and (epoch_end_callback is not None):
                epoch_end_callback(epoch, self.symbol, self.arg_params, self.aux_params)
                
            names, values = eval_metrics.get()
            for name, value in zip(names,values):
                logger.info("                     --->Epoch[%d] Train-%s=%f", epoch, name, value)
                
            # evaluation
            if eval_data:
                logger.info(" in eval process...")
                nbatch = 0
                # depth_metric = DistanceAccuracyMetric(class_names=self.class_names)
                eval_data.reset()
                eval_metrics.reset()
                self.valid_metric.reset()
                # depth_metric.reset()
                timing_results = []
                for data, fnames in eval_data:
                    nbatch += 1
                    # label_shape_det = data.label[0].shape
                    label_shape_seg = data.label[1].shape
                    self.arg_params[data_name] = mx.nd.array(data.data[0], self.ctx)
                    # self.arg_params[label_name_det] = mx.nd.array(data.label[0], self.ctx)
                    self.arg_params[label_name_seg] = mx.nd.array(data.label[1], self.ctx)
                    self.executor = self.symbol.bind(self.ctx, self.arg_params,
                        args_grad=self.grad_params, grad_req=grad_req, aux_states=self.aux_params)
                    
                    output_names = self.symbol.list_outputs()
                    output_dict = dict(zip(output_names, self.executor.outputs))

                    cpu_output_array = mx.nd.zeros(output_dict["seg_out_output"].shape)

                    ############## monitor status
                    # def stat_helper(name, array):
                    #     """wrapper for executor callback"""
                    #     import ctypes
                    #     from mxnet.ndarray import NDArray
                    #     from mxnet.base import NDArrayHandle, py_str
                    #     array = ctypes.cast(array, NDArrayHandle)
                    #     if 1:
                    #         array = NDArray(array, writable=False).asnumpy()
                    #         print (name, array.shape, np.mean(array), np.std(array),
                    #                ('%.1fms' % (float(time.time()-stat_helper.start_time)*1000)))
                    #     else:
                    #         array = NDArray(array, writable=False)
                    #         array.wait_to_read()
                    #         elapsed = float(time.time()-stat_helper.start_time)*1000.
                    #         if elapsed>5:
                    #             print (name, array.shape, ('%.1fms' % (elapsed,)))
                    #     stat_helper.start_time=time.time()
                    # stat_helper.start_time=float(time.time())
                    # self.executor.set_monitor_callback(stat_helper)
                    
                    ############## forward
                    tic = time.time()
                    self.executor.forward(is_train=True)
                    output_dict["seg_out_output"].wait_to_read()
                    timing_results.append((time.time()-tic)*1000.)
                    
                    output_dict["seg_out_output"].copyto(cpu_output_array)
                    pred_shape = output_dict["seg_out_output"].shape
                    label = mx.nd.array(data.label[1].reshape((label_shape_seg[0], label_shape_seg[1]*label_shape_seg[2])))
                    output_dict["seg_out_output"].wait_to_read()
                    seg_out_output = output_dict["seg_out_output"].asnumpy()

                    # pred_det_shape = output_dict["det_out_output"].shape
                    pred_seg_shape = output_dict["seg_out_output"].shape
                    # label_det = mx.nd.array(data.label[0].reshape((label_shape_det[0], label_shape_det[1]*label_shape_det[2])))
                    label_seg = mx.nd.array(data.label[1].reshape((label_shape_seg[0], label_shape_seg[1]*label_shape_seg[2])),ctx=self.ctx)
                    # pred_det = mx.nd.array(output_dict["det_out_output"].reshape((pred_det_shape[0], pred_det_shape[1], pred_det_shape[2])))
                    pred_seg = mx.nd.array(output_dict["seg_out_output"].reshape((pred_seg_shape[0], pred_seg_shape[1], pred_seg_shape[2]*pred_seg_shape[3])),ctx=self.ctx)

                    #### remove invalid boxes
                    # out_dets = output_dict["det_out_output"].asnumpy()
                    # assert len(out_dets.shape)==3
                    # pred_det = np.zeros((batch_size, 200, 7), np.float32)-1.
                    # for idx, out_det in enumerate(out_dets):
                    #     assert len(out_det.shape)==2
                    #     out_det = np.expand_dims(out_det, axis=0)
                    #     indices = np.where(out_det[:,:,0]>=0) # labeled as negative
                    #     out_det = np.expand_dims(out_det[indices[0],indices[1],:],axis=0)
                    #     indices = np.where(out_det[:,1]>.25) # higher confidence
                    #     out_det = np.expand_dims(out_det[indices[0],indices[1],:],axis=0)
                    #     pred_det[idx, :out_det.shape[1], :] = out_det
                    #     del out_det
                    # pred_det = mx.nd.array(pred_det)
                    
                    ##### display results
                    if False: # self.evaluation_only:
                        out_img = output_dict["seg_out_output"]
                        out_img = mx.nd.split(out_img, axis=0, num_outputs=out_img.shape[0], squeeze_axis=0)
                        if not isinstance(out_img,list):
                            out_img = [out_img]
                        for imgidx in range(eval_data.batch_size):
                            ### segmentation
                            seg_prob = out_img[imgidx]
                            seg_prob = mx.nd.array(np.squeeze(seg_prob.asnumpy(),axis=(0,)),ctx=self.ctx)
                            res_img = np.squeeze(seg_prob.asnumpy().argmax(axis=0).astype(np.uint8))
                            # res_img = np.squeeze(out_img[imgidx,:,:].argmax(axis=0).astype(np.uint8))
                            label_img = data.label[1].asnumpy()[imgidx,:,:].astype(np.uint8)
                            img = np.squeeze(data.data[0].asnumpy()[imgidx,:,:,:])
                            # det = pred_det.asnumpy()[imgidx,:,:]
                            ### ground-truth
                            # gt = label_det.asnumpy()[imgidx,:].reshape((-1,6))
                            # save to results folder for evalutation
                            res_fname = fnames[imgidx].replace("SegmentationClass","results")
                            lut = np.zeros(256)
                            # lut[:19]=np.array([7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33])
                            lut[:20]=np.array([7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33,34])
                            seg_resized = prob_upsampling(seg_prob, target_shape=(1024,2048))
                            seg_resized2 = cv2.LUT(seg_resized,lut)
                            cv2.imwrite(res_fname, seg_resized2)
                            # display result
                            display_img = display_results(res_img,np.expand_dims(label_img,axis=0),img, self.class_names)
                            res_fname = fnames[imgidx].replace("SegmentationClass","Results").replace("labelIds","results")
                            if cv2.imwrite(res_fname, display_img):
                                print(res_fname,'saved.')
                            [exit(0) if (cv2.waitKey()&0xff)==27 else None]
                        outimgiter += 1

                    if self.evaluation_only:
                        continue

                    # eval_metrics.get_metric(0).update(None,
                    #                        [output_dict["cls_prob_output"], output_dict["loc_loss_output"],
                    #                         output_dict["cls_label_output"]])
                    eval_metrics.get_metric(0).update([label_seg], [pred_seg])
                    # self.valid_metric.update([mx.nd.slice_axis(data.label[0],axis=2,begin=0,end=5)], \
                    #                          [mx.nd.slice_axis(pred_det,axis=2,begin=0,end=6)])
                    # disparities = []
                    # for imgidx in range(batch_size):
                    #     dispname = fnames[imgidx].replace("SegmentationClass","Disparity").replace("gtFine_labelTrainIds","disparity")
                    #     disparities.append(cv2.imread(dispname,-1))
                    #     assert disparities[0] is not None, dispname + " not found."
                    # depth_metric.update(mx.nd.array(disparities),[pred_det])
                    
                    # det_metric = self.valid_metric
                    seg_metric = eval_metrics.get_metric(0)
                    # det_names, det_values = det_metric.get()
                    seg_name, seg_value = seg_metric.get()
                    # depth_names, depth_values = depth_metric.get()
                    print("\r %d/%d speed=%.1fms %.1f%% %s=%.1f" % \
                          (nbatch*eval_data.batch_size,eval_data.num_samples,
                           math.fsum(timing_results)/float(nbatch),
                           float(nbatch*eval_data.batch_size)*100./float(eval_data.num_samples),
                           # det_names[-1],det_values[-1]*100.,
                           seg_name,seg_value*100.,),end='\r')
                    
                names, values = eval_metrics.get()
                for name, value in zip(names,values):
                    logger.info(' epoch[%d] Validation-%s=%f', epoch, name, value)
                logger.info('----------------------------------------------')
                print(' & '.join(names))
                print(' & '.join(map(lambda v:'%.1f'%(v*100.,),values)))
                # logger.info('----------------------------------------------')
                # names, values = self.valid_metric.get()
                # for name, value in zip(names,values):
                #     logger.info(' epoch[%d] Validation-%s=%f', epoch, name, value)
                # logger.info('----------------------------------------------')
                # print(' & '.join(names))
                # print(' & '.join(map(lambda v:'%.1f'%(v*100.,),values)))
                # logger.info('----------------------------------------------')
                # names, values = depth_metric.get()
                # for name, value in zip(names,values):
                #     logger.info(' epoch[%d] Validation-%s=%f', epoch, name, value)
                # logger.info('----------------------------------------------')
                # print(' & '.join(names))
                # print(' & '.join(map(lambda v:'%.1f'%(v*100.,),values)))
                logger.info('----------------------------------------------')
                    
                if self.evaluation_only:
                    exit(0) ## for debugging only


