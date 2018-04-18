from __future__ import print_function
import mxnet as mx
import numpy as np
from timeit import default_timer as timer
from dataset.testdb import TestDB
from dataset.iterator import DetIter
import cv2
import nms
import logging
from utils import put_text

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
        self.mod = mx.mod.Module(symbol, label_names=None, context=ctx)
        self.data_shape = data_shape
        self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape[1], data_shape[2]))])
        self.mod.set_params(args, auxs)
        self.data_shape = data_shape
        self.mean_pixels = mean_pixels

    def detect(self, det_iter, show_timer=False):
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
            if elapsed>.01:
                print (name, array.shape, ('%.1fms' % (elapsed,)))
            stat_helper.start_time=time.time()
        stat_helper.start_time=float(time.time())
        for e in self.mod._exec_group.execs:
            e.set_monitor_callback(stat_helper)
        
        start = timer()
        detections = self.mod.predict(det_iter).asnumpy()
        time_elapsed = timer() - start
        if show_timer:
            print("Detection time : {:.4f} sec".format(time_elapsed))
        result = []
        for i in range(detections.shape[0]):
            det = detections[i, :, :]
            res = det[np.where(det[:, 0] >= 0)[0]]
            result.append(res)
        return result

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
        print(test_iter.provide_data)

        ############# uncomment the following lines to visualize input image #########
        # img = np.squeeze(test_iter.getdata()[0].asnumpy())
        # img = np.swapaxes(img, 0, 2)
        # img = np.swapaxes(img, 0, 1)
        # img = (img + np.array([123.68, 116.779, 103.939]).reshape((1,1,3))).astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow("img", img)
        # if cv2.waitKey()&0xff==27: exit(0)
        
        return self.detect(test_iter, show_timer)

    def visualize_detection(self, img, dets, classes=[], thresh=0.6):
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
        import cv2
        import random
        color_white = (255, 255, 255)
        im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # change to bgr
        yscale, xscale, ch = im.shape
        color = (0,0,192)
        for det in dets:
            cls_id = int(det[0])
            bbox = [det[2]*xscale,det[3]*yscale,det[4]*xscale,det[5]*yscale]
            score = det[1]
            if score > thresh:
                bbox = map(int, bbox)
                cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=1)
                text = '%s %.0f%%' % (classes[cls_id], score*100.)
                fontFace = cv2.FONT_HERSHEY_PLAIN
                fontScale = .8
                thickness = 1
                textSize, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
                cv2.rectangle(im, (bbox[0], bbox[1]-textSize[1]), (bbox[0]+textSize[0], bbox[1]), color=(128,0,0), thickness=-1)
                cv2.putText(im, text, (bbox[0], bbox[1]),
                            color=color_white, fontFace=fontFace, fontScale=fontScale, thickness=thickness)
        cv2.imshow("result", im)

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
            dets = self.im_detect(imgname, root_dir, extension, show_timer=show_timer)
            det = dets[0]
            img = cv2.imread(imgname)
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            # idx = nms.nms(np.hstack((det[:,2:],det[:,1:2])),.7)
            # det=det[idx,:]
            self.visualize_detection(img, det, classes, thresh)
            cv2.waitKey()
        elif imgname.endswith(".mp4") or imgname.endswith(".avi") or imgname.isdigit():
            cap = cv2.VideoCapture(int(imgname) if imgname.isdigit() else imgname)
            while 1:
                _, img = cap.read()
                img, im_scale = resize(img, 600, 1024)
                img = img[32:512+32,:,:]
                dets = self.im_detect_single(img, show_timer=True)[0]
                img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
                # idx = nms.nms(np.hstack((dets[:,2:],dets[:,1:2])),.7)
                # dets=dets[idx,:]
                self.visualize_detection(img, dets, classes, thresh)
                if cv2.waitKey(1)&0xff==27:
                    break
        else:
            raise IOError("unknown file extention, only .png/.jpg/.mp4/.avi files are supported.")    
