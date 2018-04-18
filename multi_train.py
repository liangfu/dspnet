import argparse
import tools.find_mxnet
import mxnet as mx
import os, sys, re
import logging
import importlib
from dataset.iterator import MultiTaskRecordIter
from train.metric import MultiBoxMetric
from evaluate.eval_metric import MApMetric
from config.config import cfg
from symbol.multitask_symbol_factory import get_det_symbol_train, get_seg_symbol_train, get_multi_symbol_train
import cv2
from multi_solver import MultiTaskSolver
from det_solver import DetTaskSolver
from seg_solver import SegTaskSolver
from multi_init import init_from_resnet
import time
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Single-shot detection network')
    parser.add_argument('--train-path', dest='train_path', help='train record to use',
                        default=os.path.join(os.getcwd(), 'data', 'train.rec'), type=str)
    parser.add_argument('--train-list', dest='train_list', help='train list to use',
                        default="", type=str)
    parser.add_argument('--val-path', dest='val_path', help='validation record to use',
                        default=os.path.join(os.getcwd(), 'data', 'val.rec'), type=str)
    parser.add_argument('--val-list', dest='val_list', help='validation list to use',
                        default="", type=str)
    parser.add_argument('--network', dest='network', type=str, default='resnet-18',
                        help='which network to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--resume', dest='resume', type=int, default=-1,
                        help='resume training from epoch n')
    parser.add_argument('--finetune', dest='finetune', type=int, default=-1,
                        help='finetune from epoch n, rename the model before doing this')
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default=os.path.join(os.getcwd(), 'models', 'resnet-18'), type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'models', 'multitask'), type=str)
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='0', type=str)
    parser.add_argument('--begin-epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end-epoch', dest='end_epoch', help='end epoch of training',
                        default=400, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--data-shape', dest='data_shape', type=str, default="3,512,1024",
                        help='set image shape')
    parser.add_argument('--label-width', dest='label_width', type=int, default=350, 
                        help='force padding label width to sync across train and validation')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.02,
                        help='learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--wd', dest='weight_decay', type=float, default=0.0005,
                        help='weight decay')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--lr-steps', dest='lr_refactor_step', type=str, default='80, 160, 240, 320',
                        help='refactor learning rate at specified epochs')
    parser.add_argument('--lr-factor', dest='lr_refactor_ratio', type=str, default=0.5,
                        help='ratio to refactor learning rate')
    parser.add_argument('--freeze', dest='freeze_pattern', type=str, default="^(conv1_|conv2_).*",
                        help='freeze layer pattern')
    parser.add_argument('--log', dest='log_file', type=str, default="train.log",
                        help='save training log to file')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    parser.add_argument('--pattern', dest='monitor_pattern', type=str, default=".*",
                        help='monitor parameter pattern, as regex')
    parser.add_argument('--num-example', dest='num_example', type=int, default=2975,
                        help='number of image examples')
    parser.add_argument('--num-class', dest='num_class', type=int, default=12,
                        help='number of classes')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='aeroplane, bicycle, bird, boat, bottle, bus, \
                        car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                        person, pottedplant, sheep, sofa, train, tvmonitor',
                        help='string of comma separated names, or text filename')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.45,
                        help='non-maximum suppression threshold')
    parser.add_argument('--overlap', dest='overlap_thresh', type=float, default=0.5,
                        help='evaluation overlap threshold')
    parser.add_argument('--force', dest='force_nms', type=bool, default=False,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--use-difficult', dest='use_difficult', type=bool, default=False,
                        help='use difficult ground-truths in evaluation')
    parser.add_argument('--voc07', dest='use_voc07_metric', type=bool, default=False,
                        help='use PASCAL VOC 07 11-point metric')
    args = parser.parse_args()
    return args

def parse_class_names(args):
    """ parse # classes and class_names if applicable """
    num_class = args.num_class
    if len(args.class_names) > 0:
        if os.path.isfile(args.class_names):
            # try to open it to read class names
            with open(args.class_names, 'r') as f:
                class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in args.class_names.split(',')]
        assert len(class_names) == num_class, str(len(class_names))
        for name in class_names:
            assert len(name) > 0
    else:
        class_names = None
    return class_names

def putText(img, bbox, text):
    fontFace = cv2.FONT_HERSHEY_PLAIN
    fontScale = .6
    thickness = 1
    textSize, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
    cv2.rectangle(img, (bbox[0], bbox[1]-textSize[1]), (bbox[0]+textSize[0], bbox[1]), color=(0,0,0), thickness=-1)
    cv2.putText(img, text, (bbox[0], bbox[1]),
                color=(255,255,255), fontFace=fontFace, fontScale=fontScale, thickness=thickness)
    

def convert_pretrained(name, args):
    """
    Special operations need to be made due to name inconsistance, etc

    Parameters:
    ---------
    name : str
        pretrained model name
    args : dict
        loaded arguments

    Returns:
    ---------
    processed arguments as dict
    """
    return args

def get_lr_scheduler(learning_rate, lr_refactor_step, lr_refactor_ratio,
                     num_example, batch_size, begin_epoch):
    """
    Compute learning rate and refactor scheduler

    Parameters:
    ---------
    learning_rate : float
        original learning rate
    lr_refactor_step : comma separated str
        epochs to change learning rate
    lr_refactor_ratio : float
        lr *= ratio at certain steps
    num_example : int
        number of training images, used to estimate the iterations given epochs
    batch_size : int
        training batch size
    begin_epoch : int
        starting epoch

    Returns:
    ---------
    (learning_rate, mx.lr_scheduler) as tuple
    """
    assert lr_refactor_ratio > 0
    iter_refactor = [int(r) for r in lr_refactor_step.split(',') if r.strip()]
    if lr_refactor_ratio >= 1:
        return (learning_rate, None)
    else:
        lr = learning_rate
        epoch_size = num_example // batch_size
        for s in iter_refactor:
            if begin_epoch >= s:
                lr *= lr_refactor_ratio
        if lr != learning_rate:
            logging.getLogger().info("Adjusted learning rate to {} for epoch {}".format(lr, begin_epoch))
        steps = [epoch_size * (x - begin_epoch) for x in iter_refactor if x > begin_epoch]
        if not steps:
            return (lr, None)
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_refactor_ratio)
        return (lr, lr_scheduler)

def train_multitask(netname, train_path, num_classes, batch_size,
              data_shape, mean_pixels, resume, finetune, pretrained, epoch,
              prefix, ctx, begin_epoch, end_epoch, frequent, learning_rate,
              momentum, weight_decay, lr_refactor_step, lr_refactor_ratio,
              freeze_layer_pattern='',
              num_example=10000, label_pad_width=350,
              nms_thresh=0.45, force_nms=False, ovp_thresh=0.5,
              use_difficult=False, class_names=None,
              voc07_metric=False, nms_topk=400, force_suppress=False,
              train_list="", val_path="", val_list="", iter_monitor=0,
              monitor_pattern=".*", log_file=None):
    """
    Wrapper for training phase.

    Parameters:
    ----------
    netname : str
        symbol name for the network structure
    train_path : str
        record file path for training
    num_classes : int
        number of object classes, not including background
    batch_size : int
        training batch-size
    data_shape : int or tuple
        width/height as integer or (3, height, width) tuple
    mean_pixels : tuple of floats
        mean pixel values for red, green and blue
    resume : int
        resume from previous checkpoint if > 0
    finetune : int
        fine-tune from previous checkpoint if > 0
    pretrained : str
        prefix of pretrained model, including path
    epoch : int
        load epoch of either resume/finetune/pretrained model
    prefix : str
        prefix for saving checkpoints
    ctx : [mx.cpu()] or [mx.gpu(x)]
        list of mxnet contexts
    begin_epoch : int
        starting epoch for training, should be 0 if not otherwise specified
    end_epoch : int
        end epoch of training
    frequent : int
        frequency to print out training status
    learning_rate : float
        training learning rate
    momentum : float
        trainig momentum
    weight_decay : float
        training weight decay param
    lr_refactor_ratio : float
        multiplier for reducing learning rate
    lr_refactor_step : comma separated integers
        at which epoch to rescale learning rate, e.g. '30, 60, 90'
    freeze_layer_pattern : str
        regex pattern for layers need to be fixed
    num_example : int
        number of training images
    label_pad_width : int
        force padding training and validation labels to sync their label widths
    nms_thresh : float
        non-maximum suppression threshold for validation
    force_nms : boolean
        suppress overlaped objects from different classes
    train_list : str
        list file path for training, this will replace the embeded labels in record
    val_path : str
        record file path for validation
    val_list : str
        list file path for validation, this will replace the embeded labels in record
    iter_monitor : int
        monitor internal stats in networks if > 0, specified by monitor_pattern
    monitor_pattern : str
        regex pattern for monitoring network stats
    log_file : str
        log to file if enabled
    """
    # set up logger
    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join('log',time.strftime('%F-%T',time.localtime()).replace(':','-')+'.log'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # logging.basicConfig()
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # if log_file:
    #     fh = logging.FileHandler(log_file)
    #     logger.addHandler(fh)

    # check args
    if isinstance(data_shape, int):
        data_shape = (3, data_shape, data_shape)
    assert len(data_shape) == 3 and data_shape[0] == 3
    prefix += '_' + netname + '_' + str(data_shape[1])

    if isinstance(mean_pixels, (int, float)):
        mean_pixels = [mean_pixels, mean_pixels, mean_pixels]
    assert len(mean_pixels) == 3, "must provide all RGB mean values"

    logger.info(str({"train_path":train_path,"batch_size":batch_size,"data_shape":data_shape}))
    train_iter = MultiTaskRecordIter(train_path, batch_size, data_shape, mean_pixels=mean_pixels,
        label_pad_width=label_pad_width, path_imglist=train_list, enable_aug=True, **cfg.train)

    if val_path:
        val_iter = MultiTaskRecordIter(val_path, batch_size, data_shape, mean_pixels=mean_pixels,
            label_pad_width=label_pad_width, path_imglist=val_list, enable_aug=False, **cfg.valid)
    else:
        val_iter = None

    # load symbol
    logger.info(str({"num_classes":num_classes,"nms_thresh":nms_thresh,"force_suppress":force_suppress,
                     "nms_topk":nms_topk}))
    if netname in ["resnet-18", "resnet-50"]:
        net = get_fcn32s_symbol_train(netname, data_shape[1], num_classes=num_classes,
            nms_thresh=nms_thresh, force_suppress=force_suppress, nms_topk=nms_topk)
    elif netname.endswith("det"):
        net = get_det_symbol_train(netname.split("_")[0], data_shape[1], num_classes=num_classes,
            nms_thresh=nms_thresh, force_suppress=force_suppress, nms_topk=nms_topk)
    elif netname.endswith("seg"):
        net = get_seg_symbol_train(netname.split("_")[0], data_shape[1], num_classes=num_classes,
            nms_thresh=nms_thresh, force_suppress=force_suppress, nms_topk=nms_topk)
    elif netname.endswith("multi"):
        net = get_multi_symbol_train(netname.split("_")[0], data_shape[1], num_classes=num_classes,
            nms_thresh=nms_thresh, force_suppress=force_suppress, nms_topk=nms_topk)
    else:
        raise NotImplementedError("")

    ################# analyze shapes #######################
    # arg_shapes, out_shapes, aux_shapes = net.infer_shape(data=(1,3,512,1024), label_det=(1,200,6))
    # arg_names = net.list_arguments()
    # print([(n,s) for n,s in zip(arg_names,arg_shapes)])

    # define layers with fixed weight/bias
    if freeze_layer_pattern.strip():
        re_prog = re.compile(freeze_layer_pattern)
        fixed_param_names = [name for name in net.list_arguments() if re_prog.match(name)]
    else:
        fixed_param_names = None

    # load pretrained or resume from previous state
    ctx_str = '('+ ','.join([str(c) for c in ctx]) + ')'
    ctx=ctx[0]
    if resume > 0:
        logger.info("Resume training with {} from epoch {}".format(ctx_str, resume))
        _, args, auxs = mx.model.load_checkpoint(prefix, resume)
        begin_epoch = resume
        args = {key: val.as_in_context(ctx) for key, val in args.items()}
        auxs = {key: val.as_in_context(ctx) for key, val in auxs.items()}
    # elif finetune > 0:
    #     logger.info("Start finetuning with {} from epoch {}".format(ctx_str, finetune))
    #     _, args, auxs = mx.model.load_checkpoint(prefix, finetune)
    #     begin_epoch = finetune
    #     # the prediction convolution layers name starts with relu, so it's fine
    #     fixed_param_names = [name for name in net.list_arguments() if name.startswith('conv')]
    elif pretrained:
        logger.info("Start training with {} from pretrained model {}".format(ctx_str, pretrained))
        _, args, auxs = mx.model.load_checkpoint(pretrained, epoch)
        # args = convert_pretrained(pretrained, args)
        args = {key: val.as_in_context(ctx) for key, val in args.items()}
        auxs = {key: val.as_in_context(ctx) for key, val in auxs.items()}
        args, auxs = init_from_resnet(ctx, net, args, auxs)
    else:
        logger.info("Experimental: start training from scratch with {}".format(ctx_str))
        args, auxs, fixed_param_names = None, None, None
        
    # helper information
    if fixed_param_names:
        logger.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')

    # init training module
    logger.info("Creating Module ...")
    mod = mx.mod.Module(net, label_names=('label_det','seg_out_label',), logger=logger, context=ctx,
                        fixed_param_names=fixed_param_names)

    # fit parameters
    batch_end_callback = mx.callback.Speedometer(train_iter.batch_size, frequent=frequent)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    learning_rate, lr_scheduler = get_lr_scheduler(learning_rate, lr_refactor_step,
        lr_refactor_ratio, num_example, batch_size, begin_epoch)
    optimizer_params={'learning_rate':learning_rate,
                      'momentum':momentum,
                      'wd':weight_decay,
                      'lr_scheduler':lr_scheduler,
                      'clip_gradient':None,
                      'rescale_grad': 1.0 } #/ len(ctx) if len(ctx) > 0 else 1.0 }
    monitor = mx.mon.Monitor(iter_monitor, pattern=monitor_pattern) if iter_monitor > 0 else None

    # run fit net, every n epochs we run evaluation network to get mAP
    valid_metric = MApMetric(ovp_thresh, use_difficult, class_names, pred_idx=0)

    from pprint import pprint
    import numpy as np
    import cv2
    from palette import color2index, index2color

    pprint(optimizer_params)
    np.set_printoptions(formatter={"float":lambda x:"%.3f "%x},suppress=True)

    ############### uncomment the following lines to visualize network ###########################
    # dot = mx.viz.plot_network(net, shape={'data':(1,3,512,1024),"label_det":(1,200,6)})
    # dot.view()
    
    ############### uncomment the following lines to visualize data ###########################
    # data_batch, _ = train_iter.next()
    # pprint({"data":data_batch.data[0].shape,
    #         "label_det":data_batch.label[0].shape,
    #         "seg_out_label":data_batch.label[1].shape})
    # data = data_batch.data[0].asnumpy()
    # label = data_batch.label[0].asnumpy()
    # segmt = data_batch.label[1].asnumpy()
    # for ii in range(data.shape[0]):
    #     img = data[ii,:,:,:]
    #     seg = segmt[ii,:,:]
    #     print label[ii,:5,:]
    #     img = np.squeeze(img)
    #     img = np.swapaxes(img, 0, 2)
    #     img = np.swapaxes(img, 0, 1)
    #     img = (img + np.array([123.68, 116.779, 103.939]).reshape((1,1,3))).astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     rois = label[ii,:,:]
    #     hh, ww, ch = img.shape
    #     for lidx in range(rois.shape[0]):
    #         roi = rois[lidx,:]
    #         if roi[0]>=0:
    #             cv2.rectangle(img, (int(roi[1]*ww),int(roi[2]*hh)), (int(roi[3]*ww),int(roi[4]*hh)), (0,0,128))
    #             cls_id = int(roi[0])
    #             bbox = [int(roi[1]*ww),int(roi[2]*hh),int(roi[3]*ww),int(roi[4]*hh)]
    #             text = '%s %.0fm' % (class_names[cls_id], roi[5]*255.)
    #             putText(img,bbox,text)
    #     disp = np.zeros((hh*2, ww, ch),np.uint8)
    #     disp[:hh,:, :] = img.astype(np.uint8)
    #     disp[hh:,:, :] = cv2.resize(index2color(seg),(ww,hh),interpolation=cv2.INTER_NEAREST)
    #     cv2.imshow("img", disp)
    #     if cv2.waitKey()&0xff==27: exit(0)
    
    # ctx=ctx[0]
    # args = {key: val.as_in_context(ctx) for key, val in args.items()}
    # auxs = {key: val.as_in_context(ctx) for key, val in auxs.items()}
    # args, auxs = init_from_resnet(ctx, net, args, auxs)
    
    pprint({"ctx":ctx,"begin_epoch":begin_epoch,"end_epoch":end_epoch, \
           "learning_rate":learning_rate,"momentum":momentum})

    model = None
    if netname.endswith("multi"):
        model = MultiTaskSolver(
            ctx                 = ctx,
            symbol              = net,
            begin_epoch         = begin_epoch,
            num_epoch           = end_epoch, # 50 epoch
            arg_params          = args,
            aux_params          = auxs,
            learning_rate       = learning_rate, # 1e-5
            lr_scheduler        = lr_scheduler,
            momentum            = momentum,  # 0.99
            wd                  = 0.0005,     # 0.0005
            valid_metric        = valid_metric,
            class_names         = class_names,
        )
    elif netname.endswith("det"):
        model = DetTaskSolver(
            ctx                 = ctx,
            symbol              = net,
            begin_epoch         = begin_epoch,
            num_epoch           = end_epoch, # 50 epoch
            arg_params          = args,
            aux_params          = auxs,
            learning_rate       = learning_rate, # 1e-5
            lr_scheduler        = lr_scheduler,
            momentum            = momentum,  # 0.99
            wd                  = 0.0005,     # 0.0005
            valid_metric        = valid_metric,
            class_names         = class_names,
        )
    elif netname.endswith("seg"):
        model = SegTaskSolver(
            ctx                 = ctx,
            symbol              = net,
            begin_epoch         = begin_epoch,
            num_epoch           = end_epoch, # 50 epoch
            arg_params          = args,
            aux_params          = auxs,
            learning_rate       = learning_rate, # 1e-5
            lr_scheduler        = lr_scheduler,
            momentum            = momentum,  # 0.99
            wd                  = 0.0005,     # 0.0005
            valid_metric        = valid_metric,
            class_names         = class_names,
        )
    else:
        raise NotImplementedError("")
    
    model.fit(
        train_data          = train_iter,
        eval_data           = val_iter,
        batch_end_callback  = batch_end_callback,
        epoch_end_callback  = epoch_end_callback)
        
    # mod.fit(train_iter, val_iter,
    #         eval_metric=MultiBoxMetric(),
    #         validation_metric=valid_metric,
    #         batch_end_callback=batch_end_callback,
    #         epoch_end_callback=epoch_end_callback,
    #         optimizer='sgd',
    #         optimizer_params=optimizer_params,
    #         begin_epoch=begin_epoch,
    #         num_epoch=end_epoch,
    #         initializer=mx.init.Xavier(),
    #         arg_params=args,
    #         aux_params=auxs,
    #         allow_missing=True,
    #         monitor=monitor)


if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx
    # class names if applicable
    class_names = parse_class_names(args)
    # start training
    train_multitask(args.network, args.train_path,
              args.num_class, args.batch_size,
              map(lambda x:int(x),args.data_shape.split(',')),
              [args.mean_r, args.mean_g, args.mean_b],
              args.resume, args.finetune, args.pretrained,
              args.epoch, args.prefix, ctx, args.begin_epoch, args.end_epoch,
              args.frequent, args.learning_rate, args.momentum, args.weight_decay,
              args.lr_refactor_step, args.lr_refactor_ratio,
              val_path=args.val_path,
              num_example=args.num_example,
              class_names=class_names,
              label_pad_width=args.label_width,
              freeze_layer_pattern=args.freeze_pattern,
              iter_monitor=args.monitor,
              monitor_pattern=args.monitor_pattern,
              log_file=args.log_file,
              nms_thresh=args.nms_thresh,
              force_nms=args.force_nms,
              ovp_thresh=args.overlap_thresh,
              use_difficult=args.use_difficult,
              voc07_metric=args.use_voc07_metric)
