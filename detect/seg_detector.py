from __future__ import print_function
import mxnet as mx
import numpy as np
from timeit import default_timer as timer
from dataset.testdb import TestDB
from dataset.iterator import DetIter
import cv2
import nms
import logging
from palette import index2color
import time, math

short_class_name = {"traffic light":"tlight","traffic sign":"tsign","person":"person",\
                    "rider":"rider","car":"car","truck":"truck","bus":"bus","train":"train",\
                    "motorcycle":"mbike","bicycle":"bike","vegetation":"tree"}

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
        cv2.putText(annotation, name, (anchor[0]+blocksize+1, anchor[1]+10), color=(255,255,255), \
                    fontFace=fontFace, fontScale=fontScale)
    return annotation

def resize(im, target_size, max_size):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    return im, im_scale


def transform(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[i]
    return im_tensor

class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """
    def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
                 batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        print("Loading checkpoint at %s %d"%(model_prefix,epoch,))
        load_symbol, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
        if symbol is None:
            symbol = load_symbol
        self.data_shape = data_shape
        self.batch_size = batch_size
        self.method = 2
        self.imgidx = 0

        if self.method==1:
            ###### METHOD 1 - BETTER DETECTION RESULT
            self.mod = mx.mod.Module(symbol, label_names=None, context=ctx)
            self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape[1], data_shape[2]))])
            self.mod.set_params(args, auxs)
        elif self.method==2:
            ###### METHOD 2 - BETTER SEGMENTATION RESULT
            self.symbol = load_symbol
            self.args = {key: val.as_in_context(self.ctx) for key, val in args.items() if key not in ["seg_out_label"]}
            self.auxs = {key: val.as_in_context(self.ctx) for key, val in auxs.items()}
            self.args["seg_out_label"]=mx.nd.zeros(shape=(self.batch_size,self.data_shape[1]/4,self.data_shape[2]/4), \
                                                      ctx=self.ctx)
            self.args["label_det"]=mx.nd.zeros(shape=(self.batch_size,200,6),ctx=self.ctx)
        elif self.method==3:
            ###### METHOD 3 - BETTER SEGMENTATION RESULT
            self.symbol = symbol
            self.args = {key: val.as_in_context(self.ctx) for key, val in args.items()}
            self.auxs = {key: val.as_in_context(self.ctx) for key, val in auxs.items()}
        elif self.method==4:
            ###### METHOD 1 - BETTER DETECTION RESULT
            self.args = {key: val.as_in_context(self.ctx) for key, val in args.items() if key not in ["seg_out_label"]}
            self.auxs = {key: val.as_in_context(self.ctx) for key, val in auxs.items()}
            seg_out_label_shape = (self.batch_size,self.data_shape[1]/4,self.data_shape[2]/4)
            label_det_shape = (self.batch_size,200,6)
            self.args["seg_out_label"]=mx.nd.zeros(shape=(self.batch_size,self.data_shape[1]/4,self.data_shape[2]/4), \
                                                      ctx=self.ctx)
            self.args["label_det"]=mx.nd.zeros(shape=(self.batch_size,200,6),ctx=self.ctx)
            self.mod = mx.mod.Module(symbol, label_names=None, context=ctx)
            self.mod.bind(data_shapes=[('data', (self.batch_size, 3, data_shape[1], data_shape[2]))])
            # self.mod.bind(data_shapes=[('data', (self.batch_size, 3, data_shape[1], data_shape[2]))],
            #               label_shapes=[('seg_out_label',seg_out_label_shape), ("label_det",label_det_shape)])
            self.mod.set_params(args, auxs)
        
        ######## checkout argument shapes and calculate GFLOPS ########
        # gflops = 0
        # arg_shapes, out_shapes, aux_shapes = symbol.infer_shape(data=(batch_size,3,data_shape[1], data_shape[2]))
        # D=dict(zip(symbol.list_arguments(),arg_shapes))
        # internals = symbol.get_internals()
        # from utils import internal_out_shapes_512
        # out_shapes = {item[0]:item[1] for item in internal_out_shapes_512}
        # for item in D.items():
        #     if item[0].endswith("weight"):
        #         item = item[0][:-7]
        #         sym = internals[item+"_output"]
        #         wshape = D[item+"_weight"]
        #         oshape = out_shapes[item+"_output"]
        #         ops = wshape[2]*wshape[3]*oshape[2]*oshape[3]*oshape[1]*wshape[1]
        #         gflops += ops
        # print("GFLOPS: %.1f" % (gflops/(1024.*1024.*1024.),))
            
        self.data_shape = data_shape
        self.mean_pixels = mean_pixels

    def detect(self, det_iter, show_timer=False, is_image=True):
        """
        detect all images in iterator

        Parameters:
        ----------
        det_iter : DetIter
            iterator for all testing images
        show_timer : Boolean
            whether to print out detection exec time

        Returns:
        ----------
        list of detection results
        """
        # num_images = det_iter._size
        # if not isinstance(det_iter, mx.io.PrefetchingIter):
        #     det_iter = mx.io.PrefetchingIter(det_iter)

        ########## uncomment following lines to enable layer-wise timing #####################
        import time
        def stat_helper(name, array):
            """wrapper for executor callback"""
            import ctypes
            from mxnet.ndarray import NDArray
            from mxnet.base import NDArrayHandle, py_str
            array = ctypes.cast(array, NDArrayHandle)
            array = NDArray(array, writable=False)
            array.wait_to_read()
            elapsed = float(time.time()-stat_helper.start_time)*1000
            # if elapsed>0.:
            #     print (name, array.shape, ('%.1fms' % (elapsed,)))
            # stat_helper.start_time=time.time()
            array = array.asnumpy()
            print(name, array.shape, np.average(array), np.std(array), ('%.1fms' % (float(time.time()-stat_helper.start_time)*1000)))
            stat_helper.internal_shapes.append((name,array.shape))
        stat_helper.start_time=float(time.time())
        stat_helper.internal_shapes=[]
        # for e in self.mod._exec_group.execs:
        #     e.set_monitor_callback(stat_helper)
        
        start = timer()

        if self.method==1:
            ##### METHOD 1 - BETTER DETECTION RESULT
            from collections import namedtuple
            Batch = namedtuple('Batch', ['data'])
            if is_image:
                self.mod.forward(Batch([det_iter._data["data"]]))
            else:
                self.mod.forward(Batch([det_iter.data[0][1]]),is_train=True)
            results = self.mod._exec_group.execs[0].outputs
        elif self.method==2:
            ##### METHOD 2 - BETTER SEGMENTATION RESULT
            if is_image:
                self.args["data"] = det_iter._data["data"].as_in_context(self.ctx)
            else:
                self.args["data"] = det_iter.data[0][1].as_in_context(self.ctx)
            self.executor = self.symbol.bind(self.ctx, self.args, aux_states=self.auxs)
            # print('elapsed: %.1f ms' %((timer()-start)*1000.,))
            # stat_helper.start_time=float(time.time())
            # self.executor.set_monitor_callback(stat_helper)
            self.executor.forward(is_train=True)
            self.executor.outputs[0].wait_to_read()
            # self.executor.outputs[1].wait_to_read()
            # self.executor.outputs[2].wait_to_read()
            # self.executor.outputs[3].wait_to_read()
            # self.executor.outputs[4].wait_to_read()
            results = [self.executor.outputs[0],]
            # print('elapsed: %.1f ms' %((timer()-start)*1000.,))
            # print(stat_helper.internal_shapes)
        elif self.method==3:
            ##### METHOD 3 - BETTER SEGMENTATION RESULT
            if is_image:
                self.args["data"] = det_iter._data["data"].as_in_context(self.ctx)
            else:
                self.args["data"] = det_iter.data[0][1].as_in_context(self.ctx)
            self.executor = self.symbol.bind(self.ctx, self.args, aux_states=self.auxs)
            # stat_helper.start_time=float(time.time())
            # self.executor.set_monitor_callback(stat_helper)
            self.executor.forward(is_train=True)
            self.executor.outputs[0].wait_to_read()
            self.executor.outputs[1].wait_to_read()
            results = [self.executor.outputs[0],self.executor.outputs[1]]
        elif self.method==4:
            ##### METHOD 1 - BETTER DETECTION RESULT
            from collections import namedtuple
            Batch = namedtuple('Batch', ['data','label'])
            if is_image:
                self.mod.forward(Batch([det_iter._data["data"]]))
            else:
                self.mod.forward(Batch([det_iter.data[0][1]],None),is_train=False)
            # results = self.mod._exec_group.execs[0].outputs
            results = self.mod.get_outputs()
        
        # detections = results[0].asnumpy()
        # segmentation = np.squeeze(np.argmax(results[1].asnumpy(),axis=1))
        segmentation = np.squeeze(mx.nd.argmax(results[0],axis=1).asnumpy())
        time_elapsed = timer() - start
        if show_timer:
            print("Detection time : {:.4f} sec".format(time_elapsed))
        result = []
        # for i in range(detections.shape[0]):
        #     det = detections[i, :, :]
        #     res = det[np.where(det[:, 0] >= 0)[0]]
        #     result.append(res)
        return segmentation

    def im_detect(self, im_list, root_dir=None, extension=None, show_timer=False):
        """
        wrapper for detecting multiple images

        Parameters:
        ----------
        im_list : list of str
            image path or list of image paths
        root_dir : str
            directory of input images, optional if image path already
            has full directory information
        extension : str
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
        test_iter = DetIter(test_db, 1, self.data_shape, self.mean_pixels, is_train=False)

        ############# uncomment the following lines to visualize input image #########
        # img = np.squeeze(test_iter._data['data'].asnumpy())
        # img = np.swapaxes(img, 0, 2)
        # img = np.swapaxes(img, 0, 1)
        # img = (img + np.array([123.68, 116.779, 103.939]).reshape((1,1,3))).astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow("img", img)
        # if cv2.waitKey()&0xff==27: exit(0)

        return self.detect(test_iter, show_timer)

    def im_detect_single(self, img, show_timer=False):
        """
        wrapper for detecting a single image

        Parameters:
        ----------
        img : image array
            image path or list of image paths

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        data = transform(img, self.mean_pixels) # reshape to ('data', (1L, 3L, 512L, 512L))
        test_iter = mx.io.NDArrayIter(data={'data':data},label={},batch_size=1)
        # print(test_iter.provide_data)

        ############# uncomment the following lines to visualize input image #########
        # img = np.squeeze(test_iter.getdata()[0].asnumpy())
        # img = np.swapaxes(img, 0, 2)
        # img = np.swapaxes(img, 0, 1)
        # img = (img + np.array([123.68, 116.779, 103.939]).reshape((1,1,3))).astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow("img", img)
        # if cv2.waitKey()&0xff==27: exit(0)
        
        return self.detect(test_iter, show_timer, is_image=False)

    def visualize_detection(self, img, dets, seg, classes=[], thresh=0.6):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        from dataset.cs_labels import labels
        lut = np.zeros((256,3))
        for l in labels:
            if l.trainId<255 and l.trainId>=0:
                lut[l.trainId,:]=list(l.color)
        palette = lut
        # det2seg = {0:6,1:7,2:11,3:12,4:13,5:14,6:15,7:16,8:17,9:18,}
        det2seg = {0:11,1:12,2:13,3:14,4:15,5:16,6:17,7:18,}
        
        import cv2
        import random
        tic = time.time()
        color_white = (255, 255, 255)
        im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # change to bgr
        yscale, xscale, ch = im.shape
        color = (0,0,128)
        fontFace = cv2.FONT_HERSHEY_PLAIN
        fontScale = .8*(yscale/float(320))
        thickness = 2 if yscale>320 else 1
        # idx = np.argsort(dets[:,6],axis=0)[::-1] ## draw nearest first !!
        # dets = dets[idx,:]
        # for det in dets:
        #     cls_id = int(det[0])
        #     bbox = [det[2]*xscale,det[3]*yscale,det[4]*xscale,det[5]*yscale]
        #     score = det[1]
        #     distance = det[-1]
        #     if score > thresh:
        #         bbox = map(int, bbox)
        #         color = palette[det2seg[int(det[0])],(2,1,0)]
        #         cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=thickness)
        #         text = '%s %.0fm' % (short_class_name[classes[cls_id]], distance*255., )
        #         textSize, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness=1)
        #         cv2.rectangle(im, (bbox[0], bbox[1]-textSize[1]), (bbox[0]+textSize[0], bbox[1]), color=(128,0,0), thickness=-1)
        #         cv2.putText(im, text, (bbox[0], bbox[1]),
        #                     color=color_white, fontFace=fontFace, fontScale=fontScale, thickness=1)
        if seg.shape[0]!=im.shape[0]:
            seg = cv2.resize(seg, (im.shape[1],im.shape[0]), interpolation=cv2.INTER_NEAREST)
        seg = index2color(seg).astype(np.uint8)
        annotation = get_seg_labels(shape=(30, seg.shape[1], 3))
        disp = np.vstack((im,seg,annotation))
        if False: #disp.shape[1]>1000:
            hh, ww, ch = disp.shape
            resized = cv2.resize(disp, (int(round(ww*.92)),int(round(hh*.92))))
        else:
            resized = disp
        cv2.imshow("result", resized)
        # cv2.imwrite("data/cityscapes/Results/stuttgart_%06d.png" % (self.imgidx,), resized)
        # self.imgidx += 1

    def detect_and_visualize(self, imgname, root_dir=None, extension=None,
                             classes=[], thresh=0.6, show_timer=False):
        """
        wrapper for im_detect and visualize_detection

        Parameters:
        ----------
        im_list : list of str or str
            image path or list of image paths
        root_dir : str or None
            directory of input images, optional if image path already
            has full directory information
        extension : str or None
            image extension, eg. ".jpg", optional

        Returns:
        ----------

        """
        if imgname.endswith(".png") or imgname.endswith(".jpg"):
            img = cv2.imread(imgname)
            # dets, seg = self.im_detect(imgname, root_dir, extension, show_timer=show_timer)
            dets, seg = self.im_detect_single(img, show_timer=show_timer)
            det = dets[0]
            if self.data_shape[1]==320:
                img = cv2.resize(img, (640, 320))
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            # idx = nms.nms(np.hstack((det[:,2:6],det[:,1:2])),.9)
            # det=det[idx,:]
            print(det[:5,:])
            self.visualize_detection(img, det, seg, classes, thresh)
            cv2.waitKey()
        elif imgname.endswith(".mp4") or imgname.endswith(".avi") or imgname.isdigit():
            cap = cv2.VideoCapture(int(imgname) if imgname.isdigit() else imgname)
            while 1:
                tic = time.time()
                _, img = cap.read()
                if img is None: break
                img, im_scale = resize(img, 600, 1024)
                if math.fabs(float(img.shape[1])/float(img.shape[0])-2.)>.01:
                    # img = img[32:512+32,:,:]
                    img = img[32+32:512+64,:,:]
                if self.data_shape[1]==320:
                    img = cv2.resize(img, (640, 320))
                tic0 = time.time()
                seg = self.im_detect_single(img, show_timer=True)
                toc0 = time.time()
                # det = dets[0]
                img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
                # idx = nms.nms(np.hstack((det[:,2:6],det[:,1:2])),.95)
                # det=det[idx,:]
                self.visualize_detection(img, None, seg, classes, thresh)
                toc = time.time()
                # print("%.1ffps, %.1fms, %.1fms"%(1./(toc-tic),(toc0-tic0)*1000.,(toc-tic)*1000.,))
                if cv2.waitKey(1)&0xff==27:
                    break
        else:
            raise IOError("unknown file extention, only .png/.jpg/.mp4/.avi files are supported.")    
