import mxnet as mx
import numpy as np
import math

DEBUG = False

class MultiBoxMetric(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, eps=1e-8):
        super(MultiBoxMetric, self).__init__('MultiBox')
        self.eps = eps
        self.num = 2
        self.name = ['CrossEntropy', 'SmoothL1']
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        cls_prob = preds[0].asnumpy()
        loc_loss = preds[1].asnumpy()
        cls_label = preds[2].asnumpy()
        valid_count = np.sum(cls_label >= 0)
        # overall accuracy & object accuracy
        label = cls_label.flatten()
        mask = np.where(label >= 0)[0]
        indices = np.int64(label[mask])
        prob = cls_prob.transpose((0, 2, 1)).reshape((-1, cls_prob.shape[1]))
        prob = prob[mask, indices]
        self.sum_metric[0] += (-np.log(prob + self.eps)).sum()
        self.num_inst[0] += valid_count
        # smoothl1loss
        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += valid_count

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)


class CustomAccuracyMetric(mx.metric.EvalMetric):
    """Computes accuracy classification score.

    The accuracy score is defined as

    .. math::
        \\text{accuracy}(y, \\hat{y}) = \\frac{1}{n} \\sum_{i=0}^{n-1}
        \\text{1}(\\hat{y_i} == y_i)

    Parameters
    ----------
    axis : int, default=1
        The axis that represents classes
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> acc = mx.metric.CustomAccuracyMetric()
    >>> acc.update(preds = predicts, labels = labels)
    >>> print acc.get()
    ('accuracy', 0.6666666666666666)
    """
    def __init__(self, axis=1, name='accuracy',
                 output_names=None, label_names=None):
        super(CustomAccuracyMetric, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names)
        self.axis = axis

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        mx.metric.check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)

            # print(pred_label.flatten().shape,label.flatten().shape)
            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)


class DistanceAccuracyMetric(mx.metric.EvalMetric):
    """Computes accuracy classification score.

    The accuracy score is defined as

    .. math::
        \\text{accuracy}(y, \\hat{y}) = \\frac{1}{n} \\sum_{i=0}^{n-1}
        \\text{1}(\\hat{y_i} == y_i)

    Parameters
    ----------
    axis : int, default=1
        The axis that represents classes
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> acc = mx.metric.CustomAccuracyMetric()
    >>> acc.update(preds = predicts, labels = labels)
    >>> print acc.get()
    ('accuracy', 0.6666666666666666)
    """
    def __init__(self, class_names, name='derror', 
                 output_names=None, label_names=None):
        super(DistanceAccuracyMetric, self).__init__(
            name, output_names=output_names, label_names=label_names)
        self.name = class_names+[name]
        self.num = len(class_names)+1
        self.reset()
        
    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num
            self.errors = []

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        # print(labels[0].shape,preds[0].shape)
        # print(preds[0].asnumpy())
        # mx.metric.check_label_shapes(labels, preds)

        batch_size, hh, ww = labels.shape
        error = [[] for _ in range(self.num-1)]
        for label, imgs in zip(labels, preds):
            disparity = label.asnumpy()
            imgs = mx.nd.split(imgs,axis=0,num_outputs=imgs.shape[0],squeeze_axis=0)
            if not isinstance(imgs,list):
                imgs = [imgs]
            for img in imgs:
                img = np.squeeze(img.asnumpy(), axis=0)
                for bbox in img:
                    if bbox[0]<0: break
                    xmin, xmax = int(bbox[2]*ww), int(bbox[4]*ww)
                    ymin, ymax = int(bbox[3]*hh), int(bbox[5]*hh)
                    xmin, ymin = max(0,xmin), max(0,ymin)
                    if xmin==xmax: xmax=xmin+1
                    roi = disparity[ymin:ymax,xmin:xmax]
                    roi = np.squeeze(roi.reshape((1,-1)))
                    roi = roi.astype(np.float32)
                    roi = np.sort(roi)
                    if roi.shape[0]==0:
                        continue
                    dist = 2200.*75./(roi[int(math.ceil(roi.shape[0]/2))]+1e-3)
                    if dist>1000: dist = 200
                    if dist>199:
                        continue
                    error[int(bbox[0])].append(math.fabs(bbox[6]*255.-dist)/dist)
                    if DEBUG:
                        print("%.1f -> %.1f"%(dist,bbox[6]*255.,))
                if DEBUG:
                    print("---")

        for i in range(self.num-1):
            self.sum_metric[i] += math.fsum(error[i])
            self.num_inst[i] += len(error[i])
            self.errors += error[i]
        self.sum_metric[self.num-1] += math.fsum([math.fsum(error[_]) for _ in range(self.num-1)])
        self.num_inst[self.num-1] += math.fsum([len(error[_]) for _ in range(self.num-1)])

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            np.savetxt("dist_errors.txt",np.array(self.errors)*100.,fmt="%.1f")
            return (names, values)
