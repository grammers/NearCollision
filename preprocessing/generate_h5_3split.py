#!/usr/bin/python
import gc
import mat73
import cv2
import h5py
import os
import numpy as np
import scipy.io
import math


def generateH5(imgs, labels, name):
    print(len(imgs))
    name = name.split('.')[0]
    imgs = np.array(imgs)
    imgs_r = []
    imgs_u = []
    label_r = []
    label_u = []
    i = 0 
    # lopp all images
    for img in imgs:
        img_resized = cv2.resize(img.T, (224, 224)) 
        # split out every secund to achive a 5hz data set
        if i % 2 == 0:
            imgs_r.append(img_resized)
            # the nex 60 labes are for the next 6s horision
            if len(labels) > i + 60:
                label_r.append(find_label(labels[i:i+60]))
            else:
                label_r.append(find_label(labels[i:]))
        else:
            imgs_u.append(img_resized)
            if len(labels) > i + 60:
                label_u.append(find_label(labels[i:i+60]))
            else:
                label_u.append(find_label(labels[i:]))
        i += 1
    label_u = np.asarray(label_u) 
    label_r = np.asarray(label_r)
    # file with eaven indexed images
    f = h5py.File('../data/h5_file/' + name + '_e.h5', 'w')
    f.create_dataset('lable', data=label_r)
    f.create_dataset("image", data=imgs_r)
    f.close()
    # file wiht uneven indexed images
    f = h5py.File('../data/h5_file/' + name + '_u.h5', 'w')
    f.create_dataset('lable', data=label_u)
    f.create_dataset('image', data=imgs_u)
    f.close()
    print('h5 file created')

def find_label(labels):
    # value for no ner collision in set
    ret = [6.1, 6.1, 6.1]
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

if __name__ == '__main__':

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
    #target_dirs = ['../data/mats_dec/',  '../data/mats_nov/']
    target_dirs = ['../data/mats_nov/']
    #target_dirs =['../sample_data/']
    for directory in target_dirs:
        for filename in os.listdir(directory): 
            if filename.endswith(".mat"):
                labels = []

                # mats_nov contains .mat whth a diferent format
                # difrent read methods are used.
                # that cases folowup diferenses.
                #if directory == '../data/mats_dec/':
                #mat = scipy.io.loadmat(directory + filename)
                #else:
                mat = mat73.loadmat(directory + filename)
                #mat = h5py.File(directory + filename)
                print('read file')
                
                clouds = mat['clouds']
                #num = clouds[0].shape[0]
                left_img = mat['left_imgs']
                right_img = mat['right_imgs']

                mat = None
                gc.collect()

                # get the size of the images
                size_v = left_img[0][0].shape[0] #720
                size_h = left_img[0][0].shape[1] #1280
                

                ## change if not mats_nov is used
                ## for i and cloud lines
                for cloud in clouds:
                #for i in range(0, clouds[0].shape[0]):
                    #cloud = clouds[0][i]
                    
                    label  = [0, 0, 0]
                    
                    
                    # lickly problem wiht mats_dec files at the momoen
                    # check mats_dec to corect
                    #if directory == '../data/mats_now':
                    cloud = cloud.T
                    homogenized_cloud = np.ones((cloud.shape[0], 4))
                    homogenized_cloud[:,0:3] = cloud

                    image_pixels =np.matmul(intrinsic, np.matmul(new_ext, np.transpose(homogenized_cloud)))
                    image_pixels = np.divide(image_pixels, image_pixels[2,:])

                    # get desierd boundig boxes.
                    dets = np.array(
                        [[0, 0, size_v, size_h * 5 / 12],
                        [0, size_h * 5 / 12, size_v, (size_h * 7) / 12],
                        [0, (size_h * 7) / 12, size_v, size_h]])

                    for j in range(0, 3):
                        
                        y1 = int(dets[j][0])
                        x1 = int(dets[j][1])
                        y2 = int(dets[j][2])
                        x2 = int(dets[j][3])
                        

                        d = []

                        # if point is inside dounding box shoud it be considerd
                        for k in range(image_pixels.shape[1]):
                            u = int(image_pixels[0,k])
                            v = int(image_pixels[1,k])
                            # if (u > y1 & u < y2 & v > x1 & v < x2):
                            if (u > x1 and u <= x2 and v > y1 and v <= y2):
                                d.append(homogenized_cloud[k,0])


                        # cheks distens to points inside bounding boxes
                        for depth in d:
                            # near collision is defined as 1.0m
                            if depth < 1.0:
                                label[j] = 1

                    labels.append(label)

                print('filename %s' % filename)
                generateH5(left_img, labels, ('l_' + filename))
                #if directory == '../data/mats_nov2/':
                generateH5(right_img, labels, ('r_' + filename))
                exit()
