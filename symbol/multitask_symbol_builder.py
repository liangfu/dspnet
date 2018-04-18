import mxnet as mx
from common import multi_layer_feature, multibox_layer, multitask_layer
from symbol.resnet import residual_unit

eps = 2e-5
use_global_stats = False
seg_classes = 19

def import_module(module_name):
    """Helper function to import module"""
    import sys, os
    import importlib
    sys.path.append(os.path.dirname(__file__))
    return importlib.import_module(module_name)

#------------------------------------------------------------------
# SINGLE TASK: DETECTION + DEPTH ESTIMATION
#------------------------------------------------------------------

def get_det_symbol_train(network, num_classes, from_layers, num_filters, strides, pads,
                     sizes, ratios, normalizations=-1, steps=[], min_filter=128,
                     nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network symbol for training SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    label = mx.sym.Variable('label_det')
    body = import_module(network).get_symbol(num_classes, **kwargs)
    internals = body.get_internals()
    data = internals['data']
    res3 = internals[from_layers[0]+"_output"]
    res4 = internals[from_layers[1]+"_output"]
    conv_feat = internals[from_layers[2]+"_output"]

    ### remove res3 from input layer of SSD
    from_layers=from_layers[1:]
    num_filters=num_filters[1:]
    strides=strides[1:]
    pads=pads[1:]
    sizes=sizes[1:]
    ratios=ratios[1:]
    
    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter)

    loc_preds, cls_preds, anchor_boxes = multitask_layer(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    tmp = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    
    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])
    return out

def get_det_symbol(network, num_classes, from_layers, num_filters, sizes, ratios,
               strides, pads, normalizations=-1, steps=[], min_filter=128,
               nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network for testing SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    body = import_module(network).get_symbol(num_classes, **kwargs)
    internals = body.get_internals()
    data = internals['data']
    res3 = internals[from_layers[0]+"_output"]
    res4 = internals[from_layers[1]+"_output"]
    conv_feat = internals[from_layers[2]+"_output"]

    ### remove res3 from input layer of SSD
    from_layers=from_layers[1:]
    num_filters=num_filters[1:]
    strides=strides[1:]
    pads=pads[1:]
    sizes=sizes[1:]
    ratios=ratios[1:]

    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter)

    loc_preds, cls_preds, anchor_boxes = multitask_layer(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress, \
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    
    # group output
    out = mx.symbol.Group([det])
    return out

#------------------------------------------------------------------
# SINGLE TASK: SEGMENTATION
#------------------------------------------------------------------

def get_seg_symbol_train(network, num_classes, from_layers, num_filters, strides, pads,
                     sizes, ratios, normalizations=-1, steps=[], min_filter=128,
                     nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network symbol for training SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    # label = mx.sym.Variable('label_det')
    body = import_module(network).get_symbol(num_classes, **kwargs)
    internals = body.get_internals()
    data = internals['data']
    res3 = internals[from_layers[0]+"_output"]
    res4 = internals[from_layers[1]+"_output"]
    conv_feat = internals[from_layers[2]+"_output"]

    # segmentation task (pyramid pooling module)
    res3_block = mx.sym.BlockGrad(data=res3, name="res3_block")
    res3_reduced = mx.sym.Convolution(data=res3_block, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=128, no_bias=True, workspace=1024, name="res3_reduced")
    res3_reduced_bn = mx.sym.BatchNorm(data=res3_reduced, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res3_reduced_bn')
    res3_reduced2 = mx.sym.Convolution(data=res3_reduced_bn, kernel=(3,3), stride=(1,1), pad=(1,1), \
        num_filter=128, no_bias=True, workspace=1024, name="res3_reduced2")
    res3_reduced2_bn = mx.sym.BatchNorm(data=res3_reduced2, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res3_reduced2_bn')
    res4_block = mx.sym.BlockGrad(data=res4, name="res4_block")
    res4_reduced = mx.sym.Convolution(data=res4_block, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=256, no_bias=True, workspace=1024, name="res4_reduced")
    res4_reduced_bn = mx.sym.BatchNorm(data=res4_reduced, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res4_reduced_bn')
    res4_reduced2 = mx.sym.Convolution(data=res4_reduced_bn, kernel=(3,3), stride=(1,1), pad=(1,1), \
        num_filter=256, no_bias=True, workspace=1024, name="res4_reduced2")
    res4_reduced2_bn = mx.sym.BatchNorm(data=res4_reduced2, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res4_reduced2_bn')
    res5_reduced = mx.symbol.Convolution(data=conv_feat, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=512, no_bias=True, workspace=1024, name="res5_reduced")
    res5_reduced_bn = mx.sym.BatchNorm(data=conv_feat, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res5_reduced_bn')

    score_pool1 = mx.sym.Pooling(res5_reduced_bn, global_pool=False, kernel=(1,1), stride=(1,1), pad=(0,0), pool_type='avg', name='score_pool1')
    score_pool2 = mx.sym.Pooling(res5_reduced_bn, global_pool=False, kernel=(2,2), stride=(2,2), pad=(0,0), pool_type='avg', name='score_pool2')
    score_pool4 = mx.sym.Pooling(res5_reduced_bn, global_pool=False, kernel=(4,4), stride=(4,4), pad=(0,0), pool_type='avg', name='score_pool4')
    
    score2_pool4 = mx.symbol.Convolution(data=score_pool4, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=128, no_bias=True, workspace=1024, name="score2_pool4")
    score2_pool4_bn = mx.sym.BatchNorm(data=score2_pool4, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score2_pool4_bn')
    score2_pool2 = mx.symbol.Convolution(data=score_pool2, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=256, no_bias=True, workspace=1024, name="score2_pool2")
    score2_pool2_bn = mx.sym.BatchNorm(data=score2_pool2, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score2_pool2_bn')
    score2_pool1 = mx.symbol.Convolution(data=score_pool1, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=512, no_bias=True, workspace=1024, name="score2_pool1")
    score2_pool1_bn = mx.sym.BatchNorm(data=score2_pool1, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score2_pool1_bn')

    affine_matrix = mx.sym.var("affine_matrix", shape=(1,6))
    grid = mx.sym.GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(64, 128))
    score3_samp4 = mx.sym.BilinearSampler(data=score2_pool4_bn, grid=grid, name='score3_samp4')
    score3_samp2 = mx.sym.BilinearSampler(data=score2_pool2_bn, grid=grid, name='score3_samp2')
    score3_samp1 = mx.sym.BilinearSampler(data=score2_pool1_bn, grid=grid, name='score3_samp1')
    score3_sampy = mx.sym.BilinearSampler(data=res5_reduced_bn, grid=grid, name='score3_sampy')
    score3_samp0 = mx.sym.BilinearSampler(data=res4_reduced2_bn, grid=grid, name='score3_samp0')
    score3_sampx = mx.sym.BilinearSampler(data=res3_reduced2_bn, grid=grid, name='score3_sampx')
    score3_concat = mx.sym.concat(score3_samp4, score3_samp2, score3_samp1, score3_sampy, score3_samp0, score3_sampx, dim=1, name='score3_concat')
    score3_conv = mx.symbol.Convolution(data=score3_concat, kernel=(3,3), stride=(1,1), pad=(1,1), \
        num_filter=seg_classes, no_bias=True, workspace=1024, name="score3_conv")
    score3_conv_bn = mx.sym.BatchNorm(data=score3_conv, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score3_conv_bn')
    score4_conv = mx.symbol.Deconvolution(data=score3_conv_bn, kernel=(4,4), stride=(2,2), pad=(1,1), \
        num_filter=seg_classes, workspace=1024, name="score4_conv")
    fcnxs = mx.symbol.SoftmaxOutput(data=score4_conv, multi_output=True, grad_scale=4., \
        use_ignore=True, ignore_label=255, name="seg_out")
    
    # group output
    out = mx.symbol.Group([fcnxs])
    return out

def get_seg_symbol(network, num_classes, from_layers, num_filters, sizes, ratios,
               strides, pads, normalizations=-1, steps=[], min_filter=128,
               nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network for testing SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    body = import_module(network).get_symbol(num_classes, **kwargs)
    internals = body.get_internals()
    data = internals['data']
    res3 = internals[from_layers[0]+"_output"]
    res4 = internals[from_layers[1]+"_output"]
    conv_feat = internals[from_layers[2]+"_output"]

    # segmentation task (pyramid pooling module)
    res3_block = mx.sym.BlockGrad(data=res3, name="res3_block")
    res3_reduced = mx.sym.Convolution(data=res3_block, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=128, no_bias=True, workspace=1024, name="res3_reduced")
    res3_reduced_bn = mx.sym.BatchNorm(data=res3_reduced, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res3_reduced_bn')
    res3_reduced2 = mx.sym.Convolution(data=res3_reduced_bn, kernel=(3,3), stride=(1,1), pad=(1,1), \
        num_filter=128, no_bias=True, workspace=1024, name="res3_reduced2")
    res3_reduced2_bn = mx.sym.BatchNorm(data=res3_reduced2, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res3_reduced2_bn')
    res4_block = mx.sym.BlockGrad(data=res4, name="res4_block")
    res4_reduced = mx.sym.Convolution(data=res4_block, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=256, no_bias=True, workspace=1024, name="res4_reduced")
    res4_reduced_bn = mx.sym.BatchNorm(data=res4_reduced, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res4_reduced_bn')
    res4_reduced2 = mx.sym.Convolution(data=res4_reduced_bn, kernel=(3,3), stride=(1,1), pad=(1,1), \
        num_filter=256, no_bias=True, workspace=1024, name="res4_reduced2")
    res4_reduced2_bn = mx.sym.BatchNorm(data=res4_reduced2, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res4_reduced2_bn')
    res5_reduced = mx.symbol.Convolution(data=conv_feat, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=512, no_bias=True, workspace=1024, name="res5_reduced")
    res5_reduced_bn = mx.sym.BatchNorm(data=conv_feat, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res5_reduced_bn')

    score_pool1 = mx.sym.Pooling(res5_reduced_bn, global_pool=False, kernel=(1,1), stride=(1,1), pad=(0,0), pool_type='avg', name='score_pool1')
    score_pool2 = mx.sym.Pooling(res5_reduced_bn, global_pool=False, kernel=(2,2), stride=(2,2), pad=(0,0), pool_type='avg', name='score_pool2')
    score_pool4 = mx.sym.Pooling(res5_reduced_bn, global_pool=False, kernel=(4,4), stride=(4,4), pad=(0,0), pool_type='avg', name='score_pool4')
    
    score2_pool4 = mx.symbol.Convolution(data=score_pool4, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=128, no_bias=True, workspace=1024, name="score2_pool4")
    score2_pool4_bn = mx.sym.BatchNorm(data=score2_pool4, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score2_pool4_bn')
    score2_pool2 = mx.symbol.Convolution(data=score_pool2, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=256, no_bias=True, workspace=1024, name="score2_pool2")
    score2_pool2_bn = mx.sym.BatchNorm(data=score2_pool2, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score2_pool2_bn')
    score2_pool1 = mx.symbol.Convolution(data=score_pool1, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=512, no_bias=True, workspace=1024, name="score2_pool1")
    score2_pool1_bn = mx.sym.BatchNorm(data=score2_pool1, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score2_pool1_bn')

    affine_matrix = mx.sym.var("affine_matrix", shape=(1,6))
    grid = mx.sym.GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(64, 128))
    score3_samp4 = mx.sym.BilinearSampler(data=score2_pool4_bn, grid=grid, name='score3_samp4')
    score3_samp2 = mx.sym.BilinearSampler(data=score2_pool2_bn, grid=grid, name='score3_samp2')
    score3_samp1 = mx.sym.BilinearSampler(data=score2_pool1_bn, grid=grid, name='score3_samp1')
    score3_sampy = mx.sym.BilinearSampler(data=res5_reduced_bn, grid=grid, name='score3_sampy')
    score3_samp0 = mx.sym.BilinearSampler(data=res4_reduced2_bn, grid=grid, name='score3_samp0')
    score3_sampx = mx.sym.BilinearSampler(data=res3_reduced2_bn, grid=grid, name='score3_sampx')
    score3_concat = mx.sym.concat(score3_samp4, score3_samp2, score3_samp1, score3_sampy, score3_samp0, score3_sampx, dim=1, name='score3_concat')
    score3_conv = mx.symbol.Convolution(data=score3_concat, kernel=(3,3), stride=(1,1), pad=(1,1), \
        num_filter=seg_classes, no_bias=True, workspace=1024, name="score3_conv")
    score3_conv_bn = mx.sym.BatchNorm(data=score3_conv, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score3_conv_bn')
    score4_conv = mx.symbol.Deconvolution(data=score3_conv_bn, kernel=(4,4), stride=(2,2), pad=(1,1), \
        num_filter=seg_classes, workspace=1024, name="score4_conv")
    fcnxs = mx.symbol.softmax(data=score4_conv, multi_output=True, name="seg_out")
    
    # group output
    out = mx.symbol.Group([fcnxs])
    return out


#------------------------------------------------------------------
# SINGLE TASK: DETECTION + SEGMENTATION + DEPTH ESTIMATION
#------------------------------------------------------------------

def get_multi_symbol_train(network, num_classes, from_layers, num_filters, strides, pads,
                     sizes, ratios, normalizations=-1, steps=[], min_filter=128,
                     nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network symbol for training SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    label = mx.sym.Variable('label_det')
    body = import_module(network).get_symbol(num_classes, **kwargs)
    internals = body.get_internals()
    data = internals['data']
    res3 = internals[from_layers[0]+"_output"]
    res4 = internals[from_layers[1]+"_output"]
    conv_feat = internals[from_layers[2]+"_output"]

    ### remove res3 from input layer of SSD
    from_layers=from_layers[1:]
    num_filters=num_filters[1:]
    strides=strides[1:]
    pads=pads[1:]
    sizes=sizes[1:]
    ratios=ratios[1:]
    
    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter)

    loc_preds, cls_preds, anchor_boxes = multitask_layer(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    tmp = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    # segmentation task (pyramid pooling module)
    res3_block = mx.sym.BlockGrad(data=res3, name="res3_block")
    res3_reduced = mx.sym.Convolution(data=res3_block, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=128, no_bias=True, workspace=1024, name="res3_reduced")
    res3_reduced_bn = mx.sym.BatchNorm(data=res3_reduced, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res3_reduced_bn')
    res3_reduced2 = mx.sym.Convolution(data=res3_reduced_bn, kernel=(3,3), stride=(1,1), pad=(1,1), \
        num_filter=128, no_bias=True, workspace=1024, name="res3_reduced2")
    res3_reduced2_bn = mx.sym.BatchNorm(data=res3_reduced2, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res3_reduced2_bn')
    res4_block = mx.sym.BlockGrad(data=res4, name="res4_block")
    res4_reduced = mx.sym.Convolution(data=res4_block, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=256, no_bias=True, workspace=1024, name="res4_reduced")
    res4_reduced_bn = mx.sym.BatchNorm(data=res4_reduced, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res4_reduced_bn')
    res4_reduced2 = mx.sym.Convolution(data=res4_reduced_bn, kernel=(3,3), stride=(1,1), pad=(1,1), \
        num_filter=256, no_bias=True, workspace=1024, name="res4_reduced2")
    res4_reduced2_bn = mx.sym.BatchNorm(data=res4_reduced2, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res4_reduced2_bn')
    res5_reduced = mx.symbol.Convolution(data=conv_feat, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=512, no_bias=True, workspace=1024, name="res5_reduced")
    res5_reduced_bn = mx.sym.BatchNorm(data=conv_feat, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res5_reduced_bn')

    score_pool1 = mx.sym.Pooling(res5_reduced_bn, global_pool=False, kernel=(1,1), stride=(1,1), pad=(0,0), pool_type='avg', name='score_pool1')
    score_pool2 = mx.sym.Pooling(res5_reduced_bn, global_pool=False, kernel=(2,2), stride=(2,2), pad=(0,0), pool_type='avg', name='score_pool2')
    score_pool4 = mx.sym.Pooling(res5_reduced_bn, global_pool=False, kernel=(4,4), stride=(4,4), pad=(0,0), pool_type='avg', name='score_pool4')
    
    score2_pool4 = mx.symbol.Convolution(data=score_pool4, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=128, no_bias=True, workspace=1024, name="score2_pool4")
    score2_pool4_bn = mx.sym.BatchNorm(data=score2_pool4, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score2_pool4_bn')
    score2_pool2 = mx.symbol.Convolution(data=score_pool2, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=256, no_bias=True, workspace=1024, name="score2_pool2")
    score2_pool2_bn = mx.sym.BatchNorm(data=score2_pool2, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score2_pool2_bn')
    score2_pool1 = mx.symbol.Convolution(data=score_pool1, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=512, no_bias=True, workspace=1024, name="score2_pool1")
    score2_pool1_bn = mx.sym.BatchNorm(data=score2_pool1, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score2_pool1_bn')

    affine_matrix = mx.sym.var("affine_matrix", shape=(1,6))
    grid = mx.sym.GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(64, 128))
    score3_samp4 = mx.sym.BilinearSampler(data=score2_pool4_bn, grid=grid, name='score3_samp4')
    score3_samp2 = mx.sym.BilinearSampler(data=score2_pool2_bn, grid=grid, name='score3_samp2')
    score3_samp1 = mx.sym.BilinearSampler(data=score2_pool1_bn, grid=grid, name='score3_samp1')
    score3_sampy = mx.sym.BilinearSampler(data=res5_reduced_bn, grid=grid, name='score3_sampy')
    score3_samp0 = mx.sym.BilinearSampler(data=res4_reduced2_bn, grid=grid, name='score3_samp0')
    score3_sampx = mx.sym.BilinearSampler(data=res3_reduced2_bn, grid=grid, name='score3_sampx')
    score3_concat = mx.sym.concat(score3_samp4, score3_samp2, score3_samp1, score3_sampy, score3_samp0, score3_sampx, dim=1, name='score3_concat')
    score3_conv = mx.symbol.Convolution(data=score3_concat, kernel=(3,3), stride=(1,1), pad=(1,1), \
        num_filter=seg_classes, no_bias=True, workspace=1024, name="score3_conv")
    score3_conv_bn = mx.sym.BatchNorm(data=score3_conv, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score3_conv_bn')
    score4_conv = mx.symbol.Deconvolution(data=score3_conv_bn, kernel=(4,4), stride=(2,2), pad=(1,1), \
        num_filter=seg_classes, workspace=1024, name="score4_conv")
    fcnxs = mx.symbol.SoftmaxOutput(data=score4_conv, multi_output=True, grad_scale=4., \
        use_ignore=True, ignore_label=255, name="seg_out")
    
    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det, fcnxs])
    return out

def get_multi_symbol(network, num_classes, from_layers, num_filters, sizes, ratios,
               strides, pads, normalizations=-1, steps=[], min_filter=128,
               nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network for testing SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    body = import_module(network).get_symbol(num_classes, **kwargs)
    internals = body.get_internals()
    data = internals['data']
    res3 = internals[from_layers[0]+"_output"]
    res4 = internals[from_layers[1]+"_output"]
    conv_feat = internals[from_layers[2]+"_output"]

    ### remove res3 from input layer of SSD
    from_layers=from_layers[1:]
    num_filters=num_filters[1:]
    strides=strides[1:]
    pads=pads[1:]
    sizes=sizes[1:]
    ratios=ratios[1:]

    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter)

    loc_preds, cls_preds, anchor_boxes = multitask_layer(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress, \
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)

    # segmentation task (pyramid pooling module)
    res3_block = mx.sym.BlockGrad(data=res3, name="res3_block")
    res3_reduced = mx.sym.Convolution(data=res3_block, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=128, no_bias=True, workspace=1024, name="res3_reduced")
    res3_reduced_bn = mx.sym.BatchNorm(data=res3_reduced, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res3_reduced_bn')
    res3_reduced2 = mx.sym.Convolution(data=res3_reduced_bn, kernel=(3,3), stride=(1,1), pad=(1,1), \
        num_filter=128, no_bias=True, workspace=1024, name="res3_reduced2")
    res3_reduced2_bn = mx.sym.BatchNorm(data=res3_reduced2, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res3_reduced2_bn')
    res4_block = mx.sym.BlockGrad(data=res4, name="res4_block")
    res4_reduced = mx.sym.Convolution(data=res4_block, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=256, no_bias=True, workspace=1024, name="res4_reduced")
    res4_reduced_bn = mx.sym.BatchNorm(data=res4_reduced, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res4_reduced_bn')
    res4_reduced2 = mx.sym.Convolution(data=res4_reduced_bn, kernel=(3,3), stride=(1,1), pad=(1,1), \
        num_filter=256, no_bias=True, workspace=1024, name="res4_reduced2")
    res4_reduced2_bn = mx.sym.BatchNorm(data=res4_reduced2, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res4_reduced2_bn')
    res5_reduced = mx.symbol.Convolution(data=conv_feat, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=512, no_bias=True, workspace=1024, name="res5_reduced")
    res5_reduced_bn = mx.sym.BatchNorm(data=conv_feat, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='res5_reduced_bn')

    score_pool1 = mx.sym.Pooling(res5_reduced_bn, global_pool=False, kernel=(1,1), stride=(1,1), pad=(0,0), pool_type='avg', name='score_pool1')
    score_pool2 = mx.sym.Pooling(res5_reduced_bn, global_pool=False, kernel=(2,2), stride=(2,2), pad=(0,0), pool_type='avg', name='score_pool2')
    score_pool4 = mx.sym.Pooling(res5_reduced_bn, global_pool=False, kernel=(4,4), stride=(4,4), pad=(0,0), pool_type='avg', name='score_pool4')
    
    score2_pool4 = mx.symbol.Convolution(data=score_pool4, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=128, no_bias=True, workspace=1024, name="score2_pool4")
    score2_pool4_bn = mx.sym.BatchNorm(data=score2_pool4, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score2_pool4_bn')
    score2_pool2 = mx.symbol.Convolution(data=score_pool2, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=256, no_bias=True, workspace=1024, name="score2_pool2")
    score2_pool2_bn = mx.sym.BatchNorm(data=score2_pool2, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score2_pool2_bn')
    score2_pool1 = mx.symbol.Convolution(data=score_pool1, kernel=(1,1), stride=(1,1), pad=(0,0), \
        num_filter=512, no_bias=True, workspace=1024, name="score2_pool1")
    score2_pool1_bn = mx.sym.BatchNorm(data=score2_pool1, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score2_pool1_bn')

    affine_matrix = mx.sym.var("affine_matrix", shape=(1,6))
    grid = mx.sym.GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(64, 128))
    score3_samp4 = mx.sym.BilinearSampler(data=score2_pool4_bn, grid=grid, name='score3_samp4')
    score3_samp2 = mx.sym.BilinearSampler(data=score2_pool2_bn, grid=grid, name='score3_samp2')
    score3_samp1 = mx.sym.BilinearSampler(data=score2_pool1_bn, grid=grid, name='score3_samp1')
    score3_sampy = mx.sym.BilinearSampler(data=res5_reduced_bn, grid=grid, name='score3_sampy')
    score3_samp0 = mx.sym.BilinearSampler(data=res4_reduced2_bn, grid=grid, name='score3_samp0')
    score3_sampx = mx.sym.BilinearSampler(data=res3_reduced2_bn, grid=grid, name='score3_sampx')
    score3_concat = mx.sym.concat(score3_samp4, score3_samp2, score3_samp1, score3_sampy, score3_samp0, score3_sampx, dim=1, name='score3_concat')
    score3_conv = mx.symbol.Convolution(data=score3_concat, kernel=(3,3), stride=(1,1), pad=(1,1), \
        num_filter=seg_classes, no_bias=True, workspace=1024, name="score3_conv")
    score3_conv_bn = mx.sym.BatchNorm(data=score3_conv, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='score3_conv_bn')
    score4_conv = mx.symbol.Deconvolution(data=score3_conv_bn, kernel=(4,4), stride=(2,2), pad=(1,1), \
        num_filter=seg_classes, workspace=1024, name="score4_conv")
    fcnxs = mx.symbol.softmax(data=score4_conv, multi_output=True, name="seg_out")
    
    # group output
    out = mx.symbol.Group([det, fcnxs])
    return out
