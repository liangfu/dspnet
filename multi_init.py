# pylint: skip-file
import mxnet as mx
import numpy as np
import sys
import logging
from pprint import pprint
import math

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# make a bilinear interpolation kernel, return a numpy.ndarray
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1.0
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def init_from_vgg16(ctx, fcnxs_symbol, vgg16fc_args, vgg16fc_auxs):
    fcnxs_args = vgg16fc_args.copy()
    fcnxs_auxs = vgg16fc_auxs.copy()
    for k,v in fcnxs_args.items():
        if(v.context != ctx):
            fcnxs_args[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_args[k])
    for k,v in fcnxs_auxs.items():
        if(v.context != ctx):
            fcnxs_auxs[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_auxs[k])
    data_shape=(1,3,500,500)
    arg_names = fcnxs_symbol.list_arguments()
    arg_shapes, _, _ = fcnxs_symbol.infer_shape(data=data_shape)
    rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
            if x[0] in ['score_weight', 'score_bias', 'score_pool4_weight', 'score_pool4_bias', \
                        'score_pool3_weight', 'score_pool3_bias']])
    fcnxs_args.update(rest_params)
    deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes)
            if x[0] in ["bigscore_weight", 'score2_weight', 'score4_weight']])
    for k, v in deconv_params.items():
        filt = upsample_filt(v[3])
        initw = np.zeros(v)
        initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
        fcnxs_args[k] = mx.nd.array(initw, ctx)
    return fcnxs_args, fcnxs_auxs

def init_from_resnet(ctx, fcnxs_symbol, resnet_args, resnet_auxs):
    fcnxs_args = resnet_args.copy()
    fcnxs_auxs = resnet_auxs.copy()
    for k,v in fcnxs_args.items():
        if(v.context != ctx):
            fcnxs_args[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_args[k])
    for k,v in fcnxs_auxs.items():
        if(v.context != ctx):
            fcnxs_auxs[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_auxs[k])
    data_shape=(1,3,512,1024)
    arg_names = fcnxs_symbol.list_arguments()

    if 'multi_feat_2_conv_1x1_conv_weight' in arg_names:
        arg_shapes, _, _ = fcnxs_symbol.infer_shape(data=data_shape, label_det=(1,200,6))
    else:
        arg_shapes, _, _ = fcnxs_symbol.infer_shape(data=data_shape)
    
    ################### print infered shapes ######################
    # pprint(dict(zip(arg_names,arg_shapes)))
    ################### print infered shapes ######################
    fcnxs_args.update({"affine_matrix":mx.nd.array([[1,0,0,0,1,0]],ctx=ctx),})
    
    rest_params = dict([(x[0], mx.nd.random_uniform(low=-1./math.sqrt(max(x[1])),
                                                    high=1./math.sqrt(max(x[1])),shape=x[1], ctx=ctx))
                        for x in zip(arg_names, arg_shapes)
            if x[0] in ['score_weight', 'score_pool4_weight', 'score_pool3_weight', 'score_pool2_weight', 'score_pool1_weight',
                        'score2_weight', 'score2_pool4_weight', 'score2_pool3_weight', 'score2_pool2_weight', 'score2_pool1_weight',
                        'score3_conv_weight', 'score4_conv_weight',
                        "res5_reduced_weight", "res4_reduced_weight", "res3_reduced_weight", "res4_reduced2_weight", "res3_reduced2_weight", 
                        'score_shrinked_weight', 'score2_shrinked_weight',
                        '_plus5_cls_pred_conv_weight', '_plus7_cls_pred_conv_weight',
                        '_plus5_loc_pred_conv_weight', '_plus7_loc_pred_conv_weight',
                        '_plus6_cls_pred_conv_weight', '_plus12_cls_pred_conv_weight', '_plus15_cls_pred_conv_weight',
                        '_plus6_loc_pred_conv_weight', '_plus12_loc_pred_conv_weight', '_plus15_loc_pred_conv_weight',
                        'multi_feat_2_conv_1x1_conv_weight',
                        'multi_feat_2_conv_3x3_conv_weight',
                        'multi_feat_2_conv_3x3_relu_cls_pred_conv_weight',
                        'multi_feat_2_conv_3x3_relu_loc_pred_conv_weight',
                        'multi_feat_3_conv_1x1_conv_weight',
                        'multi_feat_3_conv_3x3_conv_weight',
                        'multi_feat_3_conv_3x3_relu_cls_pred_conv_weight',
                        'multi_feat_3_conv_3x3_relu_loc_pred_conv_weight',
                        'multi_feat_4_conv_1x1_conv_weight',
                        'multi_feat_4_conv_3x3_conv_weight',
                        'multi_feat_4_conv_3x3_relu_cls_pred_conv_weight',
                        'multi_feat_4_conv_3x3_relu_loc_pred_conv_weight',
                        'multi_feat_5_conv_1x1_conv_weight',
                        'multi_feat_5_conv_3x3_conv_weight',
                        'multi_feat_5_conv_3x3_relu_cls_pred_conv_weight',
                        'multi_feat_5_conv_3x3_relu_loc_pred_conv_weight',
                        'multi_feat_6_conv_1x1_conv_weight',
                        'multi_feat_6_conv_3x3_conv_weight',
                        'multi_feat_6_conv_3x3_relu_cls_pred_conv_weight',
                        'multi_feat_6_conv_3x3_relu_loc_pred_conv_weight']])
    fcnxs_args.update(rest_params)
    rest_params = dict([(x[0], mx.nd.zeros(shape=x[1], ctx=ctx)) for x in zip(arg_names, arg_shapes)
            if x[0] in ['score_bias', 'score_pool4_bias', 'score_pool3_bias', 'score_pool2_bias', 'score_pool1_bias', 
                        'score2_bias', 'score2_pool4_bias', 'score2_pool3_bias', 'score2_pool2_bias', 'score2_pool1_bias',
                        'score3_conv_bias', 'score4_conv_bias',
                        "res5_reduced_bias", "res4_reduced_bias", "res3_reduced_bias", "res4_reduced2_bias", "res3_reduced2_bias", 
                        'score_fused_bn_beta', 'score_fused_bn_bias',
                        "score_bn_beta", "score_bn_bias", 
                        "score2_bn_beta", "score2_bn_bias", 
                        "score3_conv_bn_beta", "score3_conv_bn_bias",
                        "score4_bn_beta", "score4_bn_bias",
                        'res5_reduced_bn_beta', 'res5_reduced_bn_bias', 
                        'res4_reduced_bn_beta', 'res4_reduced_bn_bias',
                        'res4_reduced2_bn_beta', 'res4_reduced2_bn_bias', 
                        'res3_reduced_bn_beta', 'res3_reduced_bn_bias', 
                        'res3_reduced2_bn_beta', 'res3_reduced2_bn_bias', 
                        "res4_bn_beta", "res4_bn_bias", 
                        "res3_bn_beta", "res3_bn_bias", 
                        "score2_pool4_bn_beta", "score2_pool4_bn_bias", 
                        "score2_pool2_bn_beta", "score2_pool2_bn_bias", 
                        "score2_pool1_bn_beta", "score2_pool1_bn_bias", 
                        '_plus5_cls_pred_conv_bias', '_plus7_cls_pred_conv_bias',
                        '_plus5_loc_pred_conv_bias', '_plus7_loc_pred_conv_bias',
                        '_plus6_cls_pred_conv_bias', '_plus12_cls_pred_conv_bias', '_plus15_cls_pred_conv_bias',
                        '_plus6_loc_pred_conv_bias', '_plus12_loc_pred_conv_bias', '_plus15_loc_pred_conv_bias',
                        'multi_feat_2_conv_1x1_conv_bias',
                        'multi_feat_2_conv_3x3_conv_bias',
                        'multi_feat_2_conv_3x3_relu_cls_pred_conv_bias',
                        'multi_feat_2_conv_3x3_relu_loc_pred_conv_bias',
                        'multi_feat_3_conv_1x1_conv_bias',
                        'multi_feat_3_conv_3x3_conv_bias',
                        'multi_feat_3_conv_3x3_relu_cls_pred_conv_bias',
                        'multi_feat_3_conv_3x3_relu_loc_pred_conv_bias',
                        'multi_feat_4_conv_1x1_conv_bias',
                        'multi_feat_4_conv_3x3_conv_bias',
                        'multi_feat_4_conv_3x3_relu_cls_pred_conv_bias',
                        'multi_feat_4_conv_3x3_relu_loc_pred_conv_bias',
                        'multi_feat_5_conv_1x1_conv_bias',
                        'multi_feat_5_conv_3x3_conv_bias',
                        'multi_feat_5_conv_3x3_relu_cls_pred_conv_bias',
                        'multi_feat_5_conv_3x3_relu_loc_pred_conv_bias',
                        'multi_feat_6_conv_1x1_conv_bias',
                        'multi_feat_6_conv_3x3_conv_bias',
                        'multi_feat_6_conv_3x3_relu_cls_pred_conv_bias',
                        'multi_feat_6_conv_3x3_relu_loc_pred_conv_bias']])
    fcnxs_args.update(rest_params)
    rest_params = dict([(x[0], mx.nd.ones(x[1], ctx)) for x in zip(arg_names, arg_shapes)
                        if x[0] in ['score_bn_gamma', 'score2_bn_gamma', 'score3_conv_bn_gamma', 'score4_bn_gamma',
                                    'res5_reduced_bn_gamma', 
                                    'res4_reduced_bn_gamma', 'res3_reduced_bn_gamma', 
                                    'res4_reduced2_bn_gamma', 'res3_reduced2_bn_gamma', 
                                    'res4_bn_gamma', 'res3_bn_gamma',
                                    'score2_pool4_bn_gamma', 'score2_pool2_bn_gamma', 'score2_pool1_bn_gamma']])
    fcnxs_args.update(rest_params)
    deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes)
                          if x[0] in ["bigscore_weight", 'score2_weight', 'score4_conv_weight']]) # , 'score3_samp4_weight', 'score3_samp2_weight', 'score3_samp1_weight'
    # pprint([(key,fcnxs_args[key]) for key in fcnxs_args.keys()])
    for k, v in deconv_params.items():
        print("Initializing %s via bilinear sampling"%(k,))
        filt = upsample_filt(v[3])
        initw = np.zeros(v)
        initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
        fcnxs_args[k] = mx.nd.array(initw, ctx)
    return fcnxs_args, fcnxs_auxs

def init_from_fcnxs(ctx, fcnxs_symbol, fcnxs_args_from, fcnxs_auxs_from):
    """ use zero initialization for better convergence, because it tends to oputut 0,
    and the label 0 stands for background, which may occupy most size of one image.
    """
    fcnxs_args = fcnxs_args_from.copy()
    fcnxs_auxs = fcnxs_auxs_from.copy()
    for k,v in fcnxs_args.items():
        if(v.context != ctx):
            fcnxs_args[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_args[k])
    for k,v in fcnxs_auxs.items():
        if(v.context != ctx):
            fcnxs_auxs[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_auxs[k])
    data_shape=(1,3,500,500)
    arg_names = fcnxs_symbol.list_arguments()
    arg_shapes, _, _ = fcnxs_symbol.infer_shape(data=data_shape)
    rest_params = {}
    deconv_params = {}
    # this is fcn8s init from fcn16s
    if 'score_pool3_weight' in arg_names:
        rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
            if x[0] in ['score_pool3_bias', 'score_pool3_weight']])
        deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes) if x[0] \
            in ["bigscore_weight", 'score4_weight']])
    # this is fcn16s init from fcn32s
    elif 'score_pool4_weight' in arg_names:
        rest_params = dict([(x[0], mx.nd.random_uniform(low=-1./math.sqrt(max(x[1])),
                                                        high=1./math.sqrt(max(x[1])),shape=x[1], ctx=ctx))
                            for x in zip(arg_names, arg_shapes)
            if x[0] in ['score_pool4_weight']] +
                           [(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
            if x[0] in ['score_pool4_bias', 'score_bn_bias', 'score_bn_beta', 'res4_bn_bias', 'res4_bn_beta']] +
                           [(x[0], mx.nd.ones(x[1], ctx)) for x in zip(arg_names, arg_shapes)
            if x[0] in ['score_bn_gamma', 'res4_bn_gamma']])
        deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes)
            if x[0] in ["bigscore_weight", 'score2_weight']])
    # this is fcn32s init
    else:
        logging.error("you are init the fcn32s model, so you should use init_from_vgg16()")
        sys.exit()
    fcnxs_args.update(rest_params)
    for k, v in deconv_params.items():
        filt = upsample_filt(v[3])
        initw = np.zeros(v)
        initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
        fcnxs_args[k] = mx.nd.array(initw, ctx)
    return fcnxs_args, fcnxs_auxs
