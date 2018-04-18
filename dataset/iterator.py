import mxnet as mx
import numpy as np
import cv2
from tools.rand_sampler import RandSampler
import os, sys
import time
np.set_printoptions(formatter={"float":lambda x:"%.3f "%x},suppress=True)
import math

class DetRecordIter(mx.io.DataIter):
    """
    The new detection iterator wrapper for mx.io.ImageDetRecordIter which is
    written in C++, it takes record file as input and runs faster.
    Supports various augment operations for object detection.

    Parameters:
    -----------
    path_imgrec : str
        path to the record file
    path_imglist : str
        path to the list file to replace the labels in record
    batch_size : int
        batch size
    data_shape : tuple
        (3, height, width)
    label_width : int
        specify the label width, use -1 for variable length
    label_pad_width : int
        labels must have same shape in batches, use -1 for automatic estimation
        in each record, otherwise force padding to width in case you want t
        rain/validation to match the same width
    label_pad_value : float
        label padding value
    resize_mode : str
        force - resize to data_shape regardless of aspect ratio
        fit - try fit to data_shape preserving aspect ratio
        shrink - shrink to data_shape only, preserving aspect ratio
    mean_pixels : list or tuple
        mean values for red/green/blue
    kwargs : dict
        see mx.io.ImageDetRecordIter

    Returns:
    ----------

    """
    def __init__(self, path_imgrec, batch_size, data_shape, path_imglist="",
                 label_width=-1, label_pad_width=-1, label_pad_value=-1,
                 resize_mode='force',  mean_pixels=[123.68, 116.779, 103.939],
                 **kwargs):
        super(DetRecordIter, self).__init__()
        self.rec = mx.io.ImageDetRecordIter(
            path_imgrec     = path_imgrec,
            path_imglist    = path_imglist,
            label_width     = label_width,
            label_pad_width = label_pad_width,
            label_pad_value = label_pad_value,
            batch_size      = batch_size,
            data_shape      = data_shape,
            mean_r          = mean_pixels[0],
            mean_g          = mean_pixels[1],
            mean_b          = mean_pixels[2],
            resize_mode     = resize_mode,
            **kwargs)

        self.provide_label = None
        self._get_batch()
        if not self.provide_label:
            raise RuntimeError("Invalid ImageDetRecordIter: " + path_imgrec)
        self.reset()

    @property
    def provide_data(self):
        return self.rec.provide_data

    def reset(self):
        self.rec.reset()

    def iter_next(self):
        return self._get_batch()

    def next(self):
        if self.iter_next():
            return self._batch
        else:
            raise StopIteration

    def _get_batch(self):
        self._batch = self.rec.next()
        if not self._batch:
            return False

        if self.provide_label is None:
            # estimate the label shape for the first batch, always reshape to n*5
            first_label = self._batch.label[0][0].asnumpy()
            self.batch_size = self._batch.label[0].shape[0]
            self.label_header_width = int(first_label[4])
            self.label_object_width = int(first_label[5])
            assert self.label_object_width >= 5, "object width must >=5"
            self.label_start = 4 + self.label_header_width
            self.max_objects = (first_label.size - self.label_start) // self.label_object_width
            self.label_shape = (self.batch_size, self.max_objects, self.label_object_width)
            self.label_end = self.label_start + self.max_objects * self.label_object_width
            self.provide_label = [('label', self.label_shape)]

        # modify label
        label = self._batch.label[0].asnumpy()
        label = label[:, self.label_start:self.label_end].reshape(
            (self.batch_size, self.max_objects, self.label_object_width))
        self._batch.label = [mx.nd.array(label)]
        return True

class DetIter(mx.io.DataIter):
    """
    Detection Iterator, which will feed data and label to network
    Optional data augmentation is performed when providing batch

    Parameters:
    ----------
    imdb : Imdb
        image database
    batch_size : int
        batch size
    data_shape : int or (int, int)
        image shape to be resized
    mean_pixels : float or float list
        [R, G, B], mean pixel values
    rand_samplers : list
        random cropping sampler list, if not specified, will
        use original image only
    rand_mirror : bool
        whether to randomly mirror input images, default False
    shuffle : bool
        whether to shuffle initial image list, default False
    rand_seed : int or None
        whether to use fixed random seed, default None
    max_crop_trial : bool
        if random crop is enabled, defines the maximum trial time
        if trial exceed this number, will give up cropping
    is_train : bool
        whether in training phase, default True, if False, labels might
        be ignored
    """
    def __init__(self, imdb, batch_size, data_shape, \
                 mean_pixels=[128, 128, 128], rand_samplers=[], \
                 rand_mirror=False, shuffle=False, rand_seed=None, \
                 is_train=True, max_crop_trial=50):
        super(DetIter, self).__init__()

        self._imdb = imdb
        self.batch_size = batch_size
        if isinstance(data_shape, int):
            data_shape = (data_shape, data_shape)
        else:
            assert len(data_shape)==3 and data_shape[0]==3
            data_shape = (data_shape[1], data_shape[2])
        self._data_shape = data_shape
        self._mean_pixels = mx.nd.array(mean_pixels).reshape((3,1,1))
        if not rand_samplers:
            self._rand_samplers = []
        else:
            if not isinstance(rand_samplers, list):
                rand_samplers = [rand_samplers]
            assert isinstance(rand_samplers[0], RandSampler), "Invalid rand sampler"
            self._rand_samplers = rand_samplers
        self.is_train = is_train
        self._rand_mirror = rand_mirror
        self._shuffle = shuffle
        if rand_seed:
            np.random.seed(rand_seed) # fix random seed
        self._max_crop_trial = max_crop_trial

        self._current = 0
        self._size = imdb.num_images
        self._index = np.arange(self._size)

        self._data = None
        self._label = None
        self._get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in self._data.items()]

    @property
    def provide_label(self):
        if self.is_train:
            return [(k, v.shape) for k, v in self._label.items()]
        else:
            return []

    def reset(self):
        self._current = 0
        if self._shuffle:
            np.random.shuffle(self._index)

    def iter_next(self):
        return self._current < self._size

    def next(self):
        if self.iter_next():
            self._get_batch()
            data_batch = mx.io.DataBatch(data=self._data.values(),
                                   label=self._label.values(),
                                   pad=self.getpad(), index=self.getindex())
            self._current += self.batch_size
            return data_batch
        else:
            raise StopIteration

    def getindex(self):
        return self._current // self.batch_size

    def getpad(self):
        pad = self._current + self.batch_size - self._size
        return 0 if pad < 0 else pad

    def _get_batch(self):
        """
        Load data/label from dataset
        """
        batch_data = mx.nd.zeros((self.batch_size, 3, self._data_shape[0], self._data_shape[1]))
        batch_label = []
        for i in range(self.batch_size):
            if (self._current + i) >= self._size:
                if not self.is_train:
                    continue
                # use padding from middle in each epoch
                idx = (self._current + i + self._size // 2) % self._size
                index = self._index[idx]
            else:
                index = self._index[self._current + i]
            # index = self.debug_index
            im_path = self._imdb.image_path_from_index(index)
            with open(im_path, 'rb') as fp:
                img_content = fp.read()
            img = mx.img.imdecode(img_content)
            gt = self._imdb.label_from_index(index).copy() if self.is_train else None
            data, label = self._data_augmentation(img, gt)
            batch_data[i] = data
            if self.is_train:
                batch_label.append(label)
        self._data = {'data': batch_data}
        if self.is_train:
            self._label = {'label': mx.nd.array(np.array(batch_label))}
        else:
            self._label = {'label': None}

    def _data_augmentation(self, data, label):
        """
        perform data augmentations: crop, mirror, resize, sub mean, swap channels...
        """
        if self.is_train and self._rand_samplers:
            rand_crops = []
            for rs in self._rand_samplers:
                rand_crops += rs.sample(label)
            num_rand_crops = len(rand_crops)
            # randomly pick up one as input data
            if num_rand_crops > 0:
                index = int(np.random.uniform(0, 1) * num_rand_crops)
                width = data.shape[1]
                height = data.shape[0]
                crop = rand_crops[index][0]
                xmin = int(crop[0] * width)
                ymin = int(crop[1] * height)
                xmax = int(crop[2] * width)
                ymax = int(crop[3] * height)
                if xmin >= 0 and ymin >= 0 and xmax <= width and ymax <= height:
                    data = mx.img.fixed_crop(data, xmin, ymin, xmax-xmin, ymax-ymin)
                else:
                    # padding mode
                    new_width = xmax - xmin
                    new_height = ymax - ymin
                    offset_x = 0 - xmin
                    offset_y = 0 - ymin
                    data_bak = data
                    data = mx.nd.full((new_height, new_width, 3), 128, dtype='uint8')
                    data[offset_y:offset_y+height, offset_x:offset_x + width, :] = data_bak
                label = rand_crops[index][1]
        if self.is_train:
            interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, \
                              cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        else:
            interp_methods = [cv2.INTER_LINEAR]
        interp_method = interp_methods[int(np.random.uniform(0, 1) * len(interp_methods))]
        data = mx.img.imresize(data, self._data_shape[1], self._data_shape[0], interp_method)
        if self.is_train and self._rand_mirror:
            if np.random.uniform(0, 1) > 0.5:
                data = mx.nd.flip(data, axis=1)
                valid_mask = np.where(label[:, 0] > -1)[0]
                tmp = 1.0 - label[valid_mask, 1]
                label[valid_mask, 1] = 1.0 - label[valid_mask, 3]
                label[valid_mask, 3] = tmp
        data = mx.nd.transpose(data, (2,0,1))
        data = data.astype('float32')
        data = data - self._mean_pixels
        return data, label



class MultiTaskRecordIter(mx.io.DataIter):
    """
    The new detection iterator wrapper for mx.io.ImageDetRecordIter which is
    written in C++, it takes record file as input and runs faster.
    Supports various augment operations for object detection.

    Parameters:
    -----------
    path_imgrec : str
        path to the record file
    path_imglist : str
        path to the list file to replace the labels in record
    batch_size : int
        batch size
    data_shape : tuple
        (3, height, width)
    label_width : int
        specify the label width, use -1 for variable length
    label_pad_width : int
        labels must have same shape in batches, use -1 for automatic estimation
        in each record, otherwise force padding to width in case you want t
        rain/validation to match the same width
    label_pad_value : float
        label padding value
    resize_mode : str
        force - resize to data_shape regardless of aspect ratio
        fit - try fit to data_shape preserving aspect ratio
        shrink - shrink to data_shape only, preserving aspect ratio
    mean_pixels : list or tuple
        mean values for red/green/blue
    kwargs : dict
        see mx.io.ImageDetRecordIter

    Returns:
    ----------

    """
    def __init__(self, path_imgrec, batch_size, data_shape, path_imglist="",
                 label_width=-1, label_pad_width=-1, label_pad_value=-1,
                 resize_mode='force',  mean_pixels=[123.68, 116.779, 103.939],
                 enable_aug=True, **kwargs):
        super(MultiTaskRecordIter, self).__init__()
        path_imgidx = path_imgrec.replace(".rec",".idx")
        path_imglst = path_imgrec.replace(".rec",".lst")
        data_shape = tuple(data_shape)
        self.enable_aug = enable_aug
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.mean_pixels = mean_pixels
        print(path_imgrec)
        print(path_imgidx)
        self.angle_range = (-5, 5)
        self.scale_range = (.5, 2.)
        self.ratio_range = (.8, 1.2)
        # self.trans_range = (.2*data_shape[2], .2*data_shape[1])
        
        # build look-up-table (LUT) for mapping segmentation label
        from dataset.cs_labels import labels
        lut = np.ones(256)*255
        for l in labels:
            if l.trainId>=0:
                lut[l.id]=l.id #trainId
        self.lut = lut

        # self.rec = mx.io.ImageDetRecordIter(
        #     path_imgrec     = path_imgrec,
        #     path_imglist    = path_imglist,
        #     label_width     = label_width,
        #     label_pad_width = label_pad_width,
        #     label_pad_value = label_pad_value,
        #     batch_size      = batch_size,
        #     data_shape      = data_shape,
        #     mean_r          = mean_pixels[0],
        #     mean_g          = mean_pixels[1],
        #     mean_b          = mean_pixels[2],
        #     resize_mode     = resize_mode,
        #     **kwargs)

        self.num_samples = sum(1 for line in open(path_imgidx,"r"))
        self.index_table = np.arange(self.num_samples)
        np.random.seed(233) # initialize random seed
        np.random.shuffle(self.index_table)
        self._reset_aug_params()
        self.curr_index = 0

        self.imglst = {}
        with open(path_imglst,"r") as fp:
            lines = fp.readlines()
            dirname = os.path.dirname(path_imglst)
            for line in lines:
                patch = line.rstrip("\n").split()
                segfile = patch[-1].replace("leftImg8bit.jpg","gtFine_labelTrainIds.png")
                segfile = segfile.replace("JPEGImages","SegmentationClass")
                self.imglst[patch[0]] = os.path.join(dirname,"cityscapes",segfile)
        
        self.rec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
        self.provide_data = [('data',[self.batch_size, self.data_shape[0], self.data_shape[1], self.data_shape[2]])]

        # self.rec = mx.io.ImageRecordIter(batch_size=batch_size, data_shape=data_shape, path_imgrec=path_imgrec, label_width=128)

        self.provide_label = None
        self._get_batch()
        if not self.provide_label:
            raise RuntimeError("Invalid ImageDetRecordIter: " + path_imgrec)
        self.reset()

    # @property
    # def provide_data(self):
    #     return self.rec.provide_data

    def reset(self):
        # self.rec.reset()
        np.random.shuffle(self.index_table)
        self.curr_index = 0
        self._reset_aug_params()

    def _reset_aug_params(self):
        self.aug_params = np.zeros((self.num_samples,6)) # flip, theta, sx, sy, tx, ty
        self.aug_params[:,0] = np.random.rand(self.num_samples)>.5 # flip
        self.aug_params[:,1] = np.radians(self.angle_range[0]+np.random.rand(self.num_samples)*(self.angle_range[1]-self.angle_range[0])) # rotate
        self.aug_params[:,2] = self.scale_range[0]+np.random.rand(self.num_samples)*(self.scale_range[1]-self.scale_range[0]) # x-scaling
        self.aug_params[:,3] = self.aug_params[:,2]*(self.ratio_range[0]+(np.random.rand(self.num_samples))*(self.ratio_range[1]-self.ratio_range[0])) # y-scaling
        self.aug_params[:,4] = -(np.random.rand(self.num_samples))*self.data_shape[2]*(self.aug_params[:,2]-1.) # tx
        self.aug_params[:,5] = -(np.random.rand(self.num_samples))*self.data_shape[1]*(self.aug_params[:,3]-1.) # ty

    def iter_next(self):
        # return self._get_batch()
        return (self.curr_index+self.batch_size) <= self.num_samples

    def next(self):
        if self.iter_next():
            # tic = time.time()
            self._get_batch()
            # print("elapsed: %.0fms" % ((time.time()-tic)*1000.,)),
            return self._batch, self._fnames
        else:
            raise StopIteration

    def _get_resized(self, img, hdr, seg, data_shape):
        hh, ww, ch = img.shape
        theta, sx, sy, tx, ty = 0., 1.*(data_shape[2]/float(ww)), 1.*(data_shape[1]/float(hh)), 0, 0
        M = np.array([[sx*math.cos(theta),-sy*math.sin(theta),tx],[sx*math.sin(theta),sy*math.cos(theta),ty]])
        img = cv2.warpAffine(img,M,(data_shape[2],data_shape[1]), flags=cv2.INTER_LINEAR)
        if seg is not None:
            seg = cv2.warpAffine(seg,M,(data_shape[2],data_shape[1]), flags=cv2.INTER_NEAREST, borderValue=(0,0,0))
        hdr_reshaped = hdr[3:].reshape((-1,6))

        ### bbox processing ###
        cls = hdr[3:].reshape((-1,6))[:,0]
        idx = np.where(cls>=0)
        pts = hdr_reshaped[:,1:5]
        dist = hdr_reshaped[idx,5]
        xop = np.ones((idx[0].shape[0],1))*self.data_shape[2]
        yop = np.ones((idx[0].shape[0],1))*self.data_shape[1]
        pts_new = pts[idx]*np.hstack((xop,yop,xop,yop))
        if pts_new.shape[0]<1:
            return img, hdr, seg
        #### remove invalid boxes ###
        xmin, ymin, xmax, ymax = hdr_reshaped[:,1],hdr_reshaped[:,2],hdr_reshaped[:,3],hdr_reshaped[:,4]
        areas = (xmax-xmin)*self.data_shape[2]*(ymax-ymin)*self.data_shape[1]
        idx_small = np.where(areas<100) # small bounding boxes
        hdr_reshaped[idx_small,:]=-1
        # idx_small = np.where(dist*255.>100) # remove objects too far away
        # hdr_reshaped[idx_small,:]=-1
        #### move annotations to top ####
        idx_valid = np.where(xmax>-.5) # removed boxes
        reshaped_top = np.squeeze(hdr_reshaped[idx_valid,:])
        hdr_reshaped.fill(-1)
        hdr_reshaped[:reshaped_top.shape[0],:] = reshaped_top
        
        #### visualize augmented results ####
        # cv2.imshow("seg",np.vstack((img,cv2.merge((seg,seg,seg)))))
        # [exit(0) if cv2.waitKey()&0xff==27 else None]
        return img, hdr, seg
    
    def _get_augmented(self, img, hdr, seg, data_shape):
        hh, ww, ch = img.shape
        aug_args = self.aug_params[self.curr_index,:]
        # print(aug_args)
        flip, theta, sx, sy, tx, ty = tuple(aug_args)
        sx2, sy2 = sx*(data_shape[2]/float(ww)), sy*(data_shape[1]/float(hh))
        M = np.array([[sx2*math.cos(theta),-sy2*math.sin(theta),tx],[sx2*math.sin(theta),sy2*math.cos(theta),ty]])
        img = cv2.warpAffine(img,M,(data_shape[2],data_shape[1]), flags=cv2.INTER_LINEAR, borderValue=(128,128,128))
        seg = cv2.warpAffine(seg,M,(data_shape[2],data_shape[1]), flags=cv2.INTER_NEAREST, borderValue=(255,255,255))
        cls = hdr[3:].reshape((-1,6))[:,0]
        idx = np.where(cls>=0)
        hdr_reshaped = hdr[3:].reshape((-1,6))
        pts = hdr_reshaped[:,1:5]
        dist = hdr_reshaped[idx,5]
        xop = np.ones((idx[0].shape[0],1))*self.data_shape[2]
        yop = np.ones((idx[0].shape[0],1))*self.data_shape[1]
        pts_new = pts[idx]*np.hstack((xop,yop,xop,yop))
        if pts_new.shape[0]<1:
            return img, hdr, seg
        
        ################## transform ##################
        pts_new = np.expand_dims(np.vstack((pts_new[:,:2],pts_new[:,2:])),axis=0)
        M = np.array([[sx*math.cos(theta),-sy*math.sin(theta),tx],[sx*math.sin(theta),sy*math.cos(theta),ty]])
        pts_new = np.squeeze(cv2.transform(pts_new,M))
        if flip>.5:
            pts_new[:,0] = self.data_shape[2]-pts_new[:,0]
        pts_new = pts_new*np.hstack((np.ones((pts_new.shape[0],1))/self.data_shape[2],
                                     np.ones((pts_new.shape[0],1))/self.data_shape[1]))
        pts_new = np.hstack((pts_new[:idx[0].shape[0],:],pts_new[idx[0].shape[0]:,:]))
        if flip>.5:
            tmp = pts_new[:,2].copy()
            pts_new[:,2] = pts_new[:,0]
            pts_new[:,0] = tmp
        #### remove invalid boxes
        xmin, ymin, xmax, ymax = pts_new[:,0],pts_new[:,1],pts_new[:,2],pts_new[:,3]
        idx_small = np.where(xmax>-.5) # removed boxes
        xmin[idx_small] = np.clip(xmin[idx_small], 0, 1)
        xmax[idx_small] = np.clip(xmax[idx_small], 0, 1)
        ymin[idx_small] = np.clip(ymin[idx_small], 0, 1)
        ymax[idx_small] = np.clip(ymax[idx_small], 0, 1)
        pts_new[:,0],pts_new[:,1],pts_new[:,2],pts_new[:,3] = xmin, ymin, xmax, ymax
        #### copy back to reshaped header
        hdr_reshaped[idx,1:5]=pts_new
        hdr_reshaped[idx,5]=dist/math.sqrt(sx*sy)
        ################## transform ##################

        #### remove invalid boxes
        xmin, ymin, xmax, ymax = hdr_reshaped[:,1],hdr_reshaped[:,2],hdr_reshaped[:,3],hdr_reshaped[:,4]
        areas = (xmax-xmin)*self.data_shape[2]*(ymax-ymin)*self.data_shape[1]
        idx_small = np.where(areas<100) # small bounding boxes
        hdr_reshaped[idx_small,:]=-1
        idx_small = np.where(xmax<.01) # out-of-image
        hdr_reshaped[idx_small,:]=-1
        idx_small = np.where(xmin>.99) # out-of-image
        hdr_reshaped[idx_small,:]=-1
        idx_small = np.where(ymax<.01) # out-of-image
        hdr_reshaped[idx_small,:]=-1
        idx_small = np.where(ymin>.99) # out-of-image
        hdr_reshaped[idx_small,:]=-1
        #### move annotations to top ####
        idx_valid = np.where(xmax>-.5) # removed boxes
        reshaped_top = np.squeeze(hdr_reshaped[idx_valid,:])
        hdr_reshaped.fill(-1)
        hdr_reshaped[:reshaped_top.shape[0],:] = reshaped_top
        
        if flip>.5:
            img = cv2.flip(img,1)
            seg = cv2.flip(seg,1)
            
        #### visualize augmented results ####
        # cv2.imshow("seg",np.vstack((img,cv2.merge((seg,seg,seg)))))
        # [exit(0) if cv2.waitKey()&0xff==27 else None]
        return img, hdr, seg
        
    def _get_batch(self):
        # self._batch = self.rec.next()

        data = np.zeros((self.batch_size, self.data_shape[0], self.data_shape[1], self.data_shape[2]))
        label = np.ones((self.batch_size, 1206))*-1
        label[:,:3] = np.array(list(self.data_shape))
        seg_out_label = mx.nd.zeros((self.batch_size,self.data_shape[1]/4,self.data_shape[2]/4))
        self._fnames = []
        for batch_idx in xrange(self.batch_size):
            item = self.rec.read_idx(self.index_table[self.curr_index])
            header, img = mx.recordio.unpack_img(item)
            hdr = np.array([header.label.shape[0]]+header.label.tolist())
            id0 = header.id
            seg = cv2.imread(self.imglst[str(id0)],-1)
            self._fnames.append(self.imglst[str(id0)])
            if self.enable_aug:
                assert seg is not None, self.imglst[str(id0)] + " not found."
                img, hdr, seg = self._get_augmented(img, hdr, seg, self.data_shape)
            else:
                img, hdr, seg = self._get_resized(img, hdr, seg, self.data_shape)
            for chidx in xrange(3):
                data[batch_idx,chidx,:,:] = img[:,:,2-chidx] - self.mean_pixels[chidx]
            if seg is not None:
                hh, ww = seg.shape
                seg = cv2.resize(seg, (ww/4,hh/4), interpolation=cv2.INTER_NEAREST)
                seg = cv2.LUT(seg,self.lut).astype(np.uint8) # apply colormap to segmentation label
                seg_out_label[batch_idx,:,:] = seg.reshape((1,self.data_shape[1]/4,self.data_shape[2]/4))
            label[batch_idx,3:3+hdr.shape[0]] = hdr
            self.curr_index += 1
        self._batch = mx.io.DataBatch(data=[mx.ndarray.array(data)], label=[mx.ndarray.array(label), seg_out_label])

        if self.provide_label is None:
            # estimate the label shape for the first batch, always reshape to n*5
            first_label = self._batch.label[0][0].asnumpy()
            print(map(int,first_label[:6]))
            print(", ".join(map(lambda x:"%.3f"%x,first_label[6:12])))
            print(", ".join(map(lambda x:"%.3f"%x,first_label[12:18])))
            self.batch_size = self._batch.label[0].shape[0]
            self.label_header_width = int(first_label[4])
            self.label_object_width = int(first_label[5])
            assert self.label_object_width >= 5, "object width must >=5"
            self.label_start = 4 + self.label_header_width
            self.max_objects = (first_label.size - self.label_start) // self.label_object_width
            self.label_shape = (self.batch_size, self.max_objects, self.label_object_width)
            self.label_end = self.label_start + self.max_objects * self.label_object_width
            self.provide_label = [('label_det', self.label_shape),('seg_out_label', seg_out_label.shape)]
            print(self.provide_label)

        # modify label
        label = self._batch.label[0].asnumpy()
        label = label[:, self.label_start:self.label_end].reshape(
            (self.batch_size, self.max_objects, self.label_object_width))
        self._batch.label = [mx.nd.array(label), seg_out_label]
        # return True
