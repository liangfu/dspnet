from __future__ import print_function
import os
import numpy as np
from imdb import Imdb
import xml.etree.ElementTree as ET
from evaluate.eval_voc import voc_eval
import cv2


class Cityscapes(Imdb):
    """
    Implementation of Imdb for Cityscapes dataset: https://www.cityscapes-dataset.com/

    Parameters:
    ----------
    anno_file : str
        annotation file for cityscapes, a json file
    image_dir : str
        image directory for cityscapes images
    shuffle : bool
        whether initially shuffle image list

    """
    def __init__(self, image_set, devkit_path, shuffle=True, is_train=False, names='cityscapes.txt'):
        # assert os.path.isfile(anno_file), "Invalid annotation file: " + anno_file
        # basename = os.path.splitext(os.path.basename(anno_file))[0]
        super(Cityscapes, self).__init__('cityscapes_' + image_set)
        self.image_set = image_set
        self.data_path = devkit_path
        self.extension = '.jpg'
        self.is_train = is_train

        self.classes = self._load_class_names(names,
            os.path.join(os.path.dirname(__file__), 'names'))

        self.config = {'use_difficult': True,
                       'comp_id': 'comp4',}

        self.num_classes = len(self.classes)
        self.image_set_index = self._load_image_set_index(shuffle)
        self.num_images = len(self.image_set_index)
        if self.is_train:
            self.labels = self._load_image_labels()

    @property
    def cache_path(self):
        """
        make a directory to store all caches

        Returns:
        ---------
            cache path
        """
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def _load_image_set_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.data_path, 'JPEGImages', name + self.extension)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        # segfile = image_file.replace("leftImg8bit.jpg","gtFine_labelIds.png")
        return image_file

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def _label_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        label_file = os.path.join(self.data_path, 'Annotations', index + '.xml')
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        return label_file

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 6] tensor
        """
        temp = []

        # load ground-truth from xml annotations
        for idx in self.image_set_index:
            label_file = self._label_path_from_index(idx)
            tree = ET.parse(label_file)
            root = tree.getroot()
            size = root.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)
            label = np.ones((200,6))*-1
            index=0

            for obj in root.iter('object'):
                # difficult = int(obj.find('difficult').text)
                # if not self.config['use_difficult'] and difficult == 1:
                #     continue
                cls_name = obj.find('name').text
                if cls_name not in self.classes:
                    continue
                cls_id = self.classes.index(cls_name)
                xml_box = obj.find('bndbox')
                xmin = float(xml_box.find('xmin').text) / width
                ymin = float(xml_box.find('ymin').text) / height
                xmax = float(xml_box.find('xmax').text) / width
                ymax = float(xml_box.find('ymax').text) / height
                distance = min(1.,max(0.,float(obj.find('distance').text)/255.))
                # label.append([cls_id, xmin, ymin, xmax, ymax, difficult])
                label[index,:]=np.array([cls_id, xmin, ymin, xmax, ymax, distance])
                index += 1
            temp.append(np.array(label))
            # print(idx)
        return temp

    def evaluate_detections(self, detections):
        """
        top level evaluations
        Parameters:
        ----------
        detections: list
            result list, each entry is a matrix of detections
        Returns:
        ----------
            None
        """
        # make all these folders for results
        result_dir = os.path.join(self.devkit_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        year_folder = os.path.join(self.devkit_path, 'results', 'VOC' + self.year)
        if not os.path.exists(year_folder):
            os.mkdir(year_folder)
        res_file_folder = os.path.join(self.devkit_path, 'results', 'VOC' + self.year, 'Main')
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        self.write_pascal_results(detections)
        self.do_python_eval()

    def get_result_file_template(self):
        """
        this is a template
        VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt

        Returns:
        ----------
            a string template
        """
        res_file_folder = os.path.join(self.devkit_path, 'results', 'VOC' + self.year, 'Main')
        comp_id = self.config['comp_id']
        filename = comp_id + '_det_' + self.image_set + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def write_pascal_results(self, all_boxes):
        """
        write results files in pascal devkit path
        Parameters:
        ----------
        all_boxes: list
            boxes to be processed [bbox, confidence]
        Returns:
        ----------
        None
        """
        for cls_ind, cls in enumerate(self.classes):
            print('Writing {} VOC results file'.format(cls))
            filename = self.get_result_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_set_index):
                    dets = all_boxes[im_ind]
                    if dets.shape[0] < 1:
                        continue
                    h, w = self._get_imsize(self.image_path_from_index(im_ind))
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        if (int(dets[k, 0]) == cls_ind):
                            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                    format(index, dets[k, 1],
                                           int(dets[k, 2] * w) + 1, int(dets[k, 3] * h) + 1,
                                           int(dets[k, 4] * w) + 1, int(dets[k, 5] * h) + 1))

    def do_python_eval(self):
        """
        python evaluation wrapper

        Returns:
        ----------
        None
        """
        annopath = os.path.join(self.data_path, 'Annotations', '{:s}.xml')
        imageset_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        cache_dir = os.path.join(self.cache_path, self.name)
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self.year) < 2010 else False
        print('VOC07 metric? ' + ('Y' if use_07_metric else 'No'))
        for cls_ind, cls in enumerate(self.classes):
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imageset_file, cls, cache_dir,
                                     ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
        print('Mean AP = {:.4f}'.format(np.mean(aps)))

    def _get_imsize(self, im_name):
        """
        get image size info
        Returns:
        ----------
        tuple of (height, width)
        """
        img = cv2.imread(im_name)
        return (img.shape[0], img.shape[1])
