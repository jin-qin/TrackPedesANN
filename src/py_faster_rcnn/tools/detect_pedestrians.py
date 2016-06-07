#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        #print i
        #print bbox[0],bbox[1],bbox[2],bbox[3]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def get_objects(dets,thresh):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    bboxs=np.zeros((len(inds),4))
    scores=np.zeros(len(inds))
    for i in inds:
        bboxs[i] = dets[i, :4]
        scores[i] = dets[i, -1]
    return {'objects':bboxs,'scores':scores}
    

def detect_objects(imgs_path,is_gpu=True):
    """

    :param imgs_path: The path of images, should be a list, e.g. # im_names = ['000456.jpg', '000542.jpg', '001150.jpg', '001763.jpg', '004545.jpg']
    :param is_gpu: Using CPU or CPU, default is GPU
    :return:
    """
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals


    prototxt = os.path.join(cfg.MODELS_DIR, NETS['vgg16'][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS['vgg16'][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if not is_gpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(0)
        cfg.GPU_ID = 0
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)

    all_objects=[]
    all_scores=[]
    if isinstance(imgs_path,str):
        detected_results = detect(net, imgs_path, 'person')
        all_objects.append(detected_results['objects'])
        all_scores.append(detected_results['scores'])
    elif isinstance(imgs_path,list):
        for im_path in imgs_path:
            print im_path
            detected_results=detect(net, im_path, 'person')
            all_objects.append(detected_results['objects'])
            all_scores.append(detected_results['scores'])
    else:
        print 'Wrong paths!'

    plt.show()
    return all_objects,all_scores

def detect(net, image_path,cls):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_path)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    cls_ind = CLASSES.index(cls)
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    #print cls_boxes
    cls_scores = scores[:, cls_ind]
    keep = np.where(cls_scores >= CONF_THRESH)[0]
    cls_boxes = cls_boxes[keep, :]
    cls_scores = cls_scores[keep]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)

    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]

    print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,
                                                                CONF_THRESH)
    vis_detections(im, cls, dets, thresh=CONF_THRESH)
    return get_objects(dets,thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('-path',dest='images_path',help='iamges path, should be a list of string')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args=parse_args()
    imgs_path=args.images_path
    print imgs_path
    all_objects,all_scores=detect_objects(imgs_path)
    np.save('detected_results',all_objects)
