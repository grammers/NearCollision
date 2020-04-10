#!/usr/bin/python
import gc
import mat73
import cv2
import h5py
import os
import numpy as np
import scipy.io
import math
import random

import sys
sys.path.insert(0, '../../tf-faster-rcnn/tools')
import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

import tensorflow as tf
from nets.resnet_v1 import resnetv1

path = '/home/grammers/catkin_ws/src/nearCollision/data/'

def target_rand():
    rand = random.random()
    if rand <= 0.2:
        target_dir = 'h5_file/duble_human_test/'
    else:
        target_dir = 'h5_file/duble_human_train/'
    return target_dir

def flip_label(label):
    for i in range(0, len(label)):
        label[i] = label[i][::-1]
    return label

def save_h5(name, split, label, imgs):
    f = h5py.File(path + target_rand() + name + split + '.h5', 'w')
    f.create_dataset('lable', data=label)
    f.create_dataset("image", data=imgs)
    f.close()

def generateH5(imgs, labels, name):
    print(len(imgs))
    name = name.split('.')[0]
    imgs_r = []
    imgs_u = []
    label_r = []
    label_u = []
    i = 0 
    # lopp all images
    for img in imgs:
        # split out every secund to achive a 5hz data set
        if i % 2 == 0:
            imgs_r.append(img)
            imgs_r.append(cv2.flip(img, 1))
            # the nex 60 labes are for the next 6s horision
            if len(labels) > i + 60:
                label_r.append(find_label(labels[i:i+60]))
            else:
                label_r.append(find_label(labels[i:]))
            label_r.append([label_r[-1][1], label_r[-1][0]])
        else:
            imgs_u.append(img)
            imgs_u.append(cv2.flip(img, 1))
            if len(labels) > i + 60:
                label_u.append(find_label(labels[i:i+60]))
            else:
                label_u.append(find_label(labels[i:]))
            label_u.append([label_u[-1][1], label_u[-1][0]])
        i += 1
    label_ua = np.asarray(label_u) 
    label_ra = np.asarray(label_r)
    # file with eaven indexed images
    save_h5(name, '_e', label_ra, imgs_r)
    save_h5(name, '_u', label_ua, imgs_u) 
    '''
    label_fu = np.asarray(flip_label(label_u))
    label_fr = np.asarray(flip_label(label_r))
    imgs_r = fliper(imgs_r)
    imgs_u = fliper(imgs_u)

    save_h5(name, '_e_f', label_fr, imgs_r)
    save_h5(name, '_u_f', label_fu, imgs_u)
    '''
    print('h5 file created')

def find_label(labels):
    # value for no ner collision in set
    ret = [6.1, 6.1]
    i = 0.0
    for label in labels:
        j = 0
        for l in label:
            if l == 1:
                if ret[j] == 6.1:
                    # every i is 0.1s
                    ret[j] = float(i * 0.1)
            j += 1
        i += 1.0
    return ret

def fliper(imgs):
    r_img = []
    for img in imgs:
        r_img.append(cv2.flip(img, 1))
    return r_img

def reSize(images, is_nov):
    imgs = np.array(images)
    r_img = []
    for img in imgs:
        if is_nov:
            r_img.append(cv2.resize(img.T, (224, 224)))
        else:
            r_img.append(cv2.resize(img, (224, 224)))
    return r_img

if __name__ == '__main__':

    cfg.TEST.HAS_RPN = True

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    #inint sesion
    sess = tf.Session(config=tfconfig)
    
    tfmodel = path + 'trained_models/voc_2007_trainval/res101_faster_rcnn_iter_70000.ckpt'
    # load box network
    net = resnetv1(num_layers=101)
    net.create_architecture("TEST", 21,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    extrinsic = np.array(
        [[-0.0077295, -0.9995, -0.030778, -0.075815], 
        [-0.031705, 0.031008, -0.99902, -0.11444],
        [0.99947, -0.0067461, -0.031928, -0.0041746]])
   
    intrinsic = np.array(
        [[6.5717649994077067e+02, 0, 6.7283525424752838e+02],
        [0, 6.5708440653270486e+02, 3.9887849408959755e+02],
        [0, 0, 1 ]])

    new_ext = extrinsic

    # raw files plased in difrent maps
    target_dirs = ['../data/mats_dec/',  '../data/mats_nov/']
    #target_dirs = ['../data/mats_nov/']
    #target_dirs =['../sample_data/']
    for directory in target_dirs:
        for filename in os.listdir(directory): 
            if filename.endswith(".mat"):

                # mats_nov contains .mat whth a diferent format
                # difrent read methods are used.
                # that cases folowup diferenses.
                if directory == '../data/mats_nov/':
                    mat = mat73.loadmat(directory + filename)
                else:
                    mat = scipy.io.loadmat(directory + filename)
                #mat = h5py.File(directory + filename)
                print('read file')
                
                clouds = mat['clouds']
                #num = clouds[0].shape[0]
                left_img = mat['left_imgs']
                right_img = mat['right_imgs']

                mat = None
                gc.collect()

                
                #imgs = np.array(left_img)
                if directory == '../data/mats_nov/':
                    left_r_img = reSize(left_img, True)
                    right_r_img = reSize(right_img, True)
                else:
                    left_r_img = reSize(left_img[0], False)
                    right_r_img = reSize(right_img[0], False)

                # get the size of the images
                size_v = left_img[0][0].shape[0] # original 720
                size_h = left_img[0][0].shape[1] #1280
                
                #for img in imgs:
                #    print(img[0].shape)
                #    img_resized = cv2.resize(img[0], (224, 224)) 
                
                #    scores, boxes = im_detect(sess, net, img[0])
                #    print(scores)
                #    print(boxes)
                #    exit()

                ## change if not mats_nov is used
                ## for i and cloud lines
                #i = 0
                #for cloud in clouds:

                labels = []

                if directory == '../data/mats_nov/':
                    r = len(clouds)
                else:
                    r = clouds[0].shape[0]
                for i in range(0, r):
                    if directory == '../data/mats_nov/':
                        cloud = clouds[i]
                        cloud = cloud.T
                    else:
                        cloud = clouds[0][i]
                    
                    label  = [0, 0]
                    
                    
                    # lickly problem wiht mats_dec files at the momoen
                    # check mats_dec to corect
                    homogenized_cloud = np.ones((cloud.shape[0], 4))
                    homogenized_cloud[:,0:3] = cloud

                    image_pixels =np.matmul(intrinsic, np.matmul(new_ext, np.transpose(homogenized_cloud)))
                    image_pixels = np.divide(image_pixels, image_pixels[2,:])
                    
                    scores, boxes = im_detect(sess, net, left_r_img[i])
                    # Visualize detections for each class
                    CONF_THRESH = 0.8 ## Only for pedestrian 
                    NMS_THRESH = 0.3

                    dets_out = []
                    n = filename[0]

                    # for cls_ind, cls in enumerate(CLASSES[1:]):
                        # cls_ind += 1 # because we skipped background

                    cls_ind = 15 
                    cls = 'person'
                    
                    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
                    cls_scores = scores[:, cls_ind]
                    dets = np.hstack((cls_boxes,
                                      cls_scores[:, np.newaxis])).astype(np.float32)
                    keep = nms(dets, NMS_THRESH)
                    dets = dets[keep, :]
                    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

                    if len(inds) > 0:
                        dets_out.extend(dets[inds, :])

                    dets_out = np.array(dets_out)

                    # get desierd boundig boxes.
                    #dets = np.array(
                    #    [[0, 0, size_v, size_h * 5 / 12],
                    #    [0, size_h * 5 / 12, size_v, (size_h * 7) / 12],
                    #    [0, (size_h * 7) / 12, size_v, size_h]])
                    dets = np.array([[0, 0, size_v, size_h / 2],
                        [0, size_h / 2, size_v, size_h]])

                    for j in range(0, len(dets)):
                        
                        y1 = int(dets[j][0])
                        x1 = int(dets[j][1])
                        y2 = int(dets[j][2])
                        x2 = int(dets[j][3])


                        for dets_b in dets_out:

                            d = []
                            Y1 = int(dets_b[0])
                            X1 = int(dets_b[1])
                            Y2 = int(dets_b[2])
                            X2 = int(dets_b[3])

                            # if point is inside dounding box shoud it be considerd
                            for k in range(image_pixels.shape[1]):
                                u = int(image_pixels[0,k])
                                v = int(image_pixels[1,k])
                                # if (u > y1 & u < y2 & v > x1 & v < x2):
                                if (u > x1 and u <= x2 and v > y1 and v <= y2):
                                    if (u > X1 and u <= X2 and v > Y1 and v <= Y2):
                                        d.append(homogenized_cloud[k,0])


                            # cheks distens to points inside bounding boxes
                            depth = np.median(d)
                            if depth < 1.1:
                                label[j] = 1

                    labels.append(label)

                print('filename %s' % filename)
                generateH5(left_r_img, labels, ('l_' + filename))
                #if directory == '../data/mats_nov2/':
                generateH5(right_r_img, labels, ('r_' + filename))
