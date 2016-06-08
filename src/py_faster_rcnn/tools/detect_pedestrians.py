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
from PIL import Image
from test_camera import *
import thread
from multiprocessing.pool import ThreadPool
import threading

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

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    return im

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
    

def setup_model(is_gpu=True):
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
    return net

def detect_objects(img,net,is_gpu=True):
    """
    Detect the pedestrians from image
    :param img: input image (OpenCV format)
    :param net: Get from function: setup_model()
    :param is_gpu:  Using GPU or not
    :return: detected_img,all_objects,all_scores
    """

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)

    all_objects=[]
    all_scores=[]
    detected_img,detected_results = detect(net, img, 'person')
    if detected_results!=None:
        all_objects.append(detected_results['objects'])
        all_scores.append(detected_results['scores'])
    else:
        all_objects=[]
        all_scores=[]
    return detected_img,all_objects,all_scores

def detect(net, im,cls):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im = cv2.imread(image_path)

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

    cls_scores = scores[:, cls_ind]
    keep = np.where(cls_scores >= CONF_THRESH)[0]
    cls_boxes = cls_boxes[keep, :]
    cls_scores = cls_scores[keep]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)

    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]

    print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,CONF_THRESH)

    if list(dets)!=[]:
        detected_img=vis_detections(im, cls, dets, thresh=CONF_THRESH)
    else:
        detected_img=im
    return detected_img,get_objects(dets,thresh=CONF_THRESH)

def detect_objects_from_video(video_path,is_gpu=True):
    cap=cv2.VideoCapture(video_path)
    print video_path
    net=setup_model()
    video_writer = cv2.VideoWriter('test_video.avi',cv2.cv.CV_FOURCC('X','V','I','D'),24,(640,480))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detected_img=detect_objects(frame,net)
        video_writer.write(detected_img[0])
    video_writer.release()


def detect_objects_from_camera():
    camera=open_camera()
    net = setup_model()
    cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    #cv2.startWindowThread()
    #pool = ThreadPool(processes=1)
    #async_result = pool.apply_async(thread_show_camera_frames, (camera,))
    #thread.start_new_thread(thread_show_camera_frames,(camera,))
    while True:
        retval, frame = camera.read()
        if not retval:
            print "Read failed!"
            exit(-1)
        detected_results = detect_objects(frame, net)
        cv2.imshow('test', detected_results[0])
        if cv2.waitKey(1) == 27:
            print "Press ESC to exit"
            break
    cv2.destroyAllWindows()
def thread_show_camera_frames(camera):
    cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    while True:
        retval, frame = camera.read()
        if not retval:
            print "Read failed!"
        cv2.imshow('test', frame)
        if cv2.waitKey(1) == 27:
            print "Press ESC to exit"
            break
    cv2.destroyAllWindows()

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
    #args=parse_args()
    #imgs_path=args.images_path
    #print imgs_path
    #all_objects,all_scores=detect_objects(imgs_path)
    #np.save('detected_results',all_objects)
    #detect_objects_from_video('/home/jin/Desktop/TrackingCNN/src/data/caltech/set00/V000.seq')
    detect_objects_from_camera()