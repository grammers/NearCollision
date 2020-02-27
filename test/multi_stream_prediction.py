#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import subprocess

import cv2

import rospy 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#from utils.timer import Timer
#import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

#from nets.vgg16 import vgg16
#from nets.resnet_v1 import resnetv1


##### Imports for prediction network #####
import torch
import torchvision.models as models
#import h5py 
#from logger import Logger
from torchvision.transforms import transforms 
import torch.utils.data as data
import numpy as np 
import pdb
import torch.nn as nn 
import torch.optim as optim 
from torch.autograd import Variable
import shutil
import os 
import random
import torch.nn.functional as F
##########################################

## latency 
import timeit 

## For multiple frames 
from collections import deque 

## For publishing time 
from std_msgs.msg import Float32 
##########################################


CLASSES = ('__background__',
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

##########################################
vgg_model = models.vgg16(pretrained=False)
num_final_in = vgg_model.classifier[-1].in_features
NUM_CLASSES = 20 
vgg_model.classifier[-1] = nn.Linear(num_final_in, NUM_CLASSES)
model_path = '/home/hexa/catkin_workspaces/catkin_samuel/src/nearCollision/data/trained_models/vgg_on_voc800'
vgg_model.load_state_dict(torch.load(model_path)) 
##########################################

class MultiStreamNearCollision(nn.Module):

    def __init__(self):

        super(MultiStreamNearCollision, self).__init__()

        self.rgb_net = self.get_vgg_features()

        kernel_size = 3 
        padding = int((kernel_size - 1)/2)
        self.conv_layer = nn.Conv2d(512, 16, kernel_size, 1, padding, bias=True)
        self.feature_size = 4704 

        self.final_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feature_size, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048,1)
        )

	## spatial_features - list of multiple img features  
    def forward(self, rgb):
        four_imgs = []
        for i in range(rgb.shape[1]):
            img_features = self.rgb_net(rgb[:,i,:,:,:])
            channels_reduced = self.conv_layer(img_features)
            img_features = channels_reduced.view((-1, 16*7*7))
            four_imgs.append(img_features)
        concat_output = torch.cat(four_imgs, dim = 1)
        out = self.final_layer(concat_output)
        return out


    def preprocess(self, cv_image):

        ## Resize 
        r_img = cv2.resize(cv_image, (224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        rgb = transform(r_img)

        ## unsqueeze it to become torch.Size([1, 3, 224, 224])
        rgb = rgb.unsqueeze(0)
        rgb = rgb.float().cuda()

        return rgb 

    def get_vgg_features(self):

        modules = list(vgg_model.children())[:-1]
        vgg16 = nn.Sequential(*modules)

        return vgg16.type(torch.Tensor) 


class image_converter:

    def __init__(self):

        ## Object of predictNearCollision class will be created here 
        #self.predNet = predictNearCollision()
        self.nstream = MultiStreamNearCollision().cuda()
        self.nstream.eval()

		#self.nstream.load_state_dict(torch.load('../../../data/model_files/4Image6s_004'))
        self.nstream.load_state_dict(torch.load('/home/hexa/catkin_workspaces/catkin_samuel/src/nearCollision/data/trained_models/6Image6s_027'))

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)

        self.counter = 0 ## Intializing a counter, alternately I can initialize a queue 

        self.stack_imgs = deque(maxlen=6)   ## 4 frames 

	## To check the frequency 
	#self.image_pub = rospy.Publisher("image_topic_2", Image)
        self.time_pub = rospy.Publisher('near_collision_time', Float32, queue_size = 10)

    def parse_args(self):
        """Parse input arguments."""
        parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
        parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]', choices=NETS.keys(), default='res101')
        parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]', choices=DATASETS.keys(), default='pascal_voc_0712')
        args = parser.parse_args()

        return args

    '''
    def vis_detections(self, im, class_name, dets, time, thresh=0.5):
		"""Draw detected bounding boxes."""
		inds = np.where(dets[:, -1] >= thresh)[0]
		if len(inds) == 0:
			return


		for i in inds:
			bbox = dets[i, :4]
			score = dets[i, -1]

			cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), 
				(int(bbox[2]), int(bbox[3])),
				(255, 255, 0), 2)

			cv2.putText(im,  str(time) + " s", (int(bbox[0]), int(bbox[1]-2)), 
				cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3) ## thickness = 3, blue color 


		#cv2.imshow("Image window", im)
		cv2.imwrite('predictions/'+str(self.counter)+'.png', im)
		#cv2.waitKey(1)
        '''

    def callback(self,data):

        #pdb.set_trace() ## Trace to see the GPU allocation 
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        #cv_image = data
	## cv_image is my im
        
        #cv_image = tf.convert_to_tensor(data.data, dtype=tf.uint8)

        # # Visualize detections for each class 
        CONF_THRESH = 0.8 
        NMS_THRESH = 0.3 

        ## cv_image is my im to be passed into the network


        input_imgs = self.nstream.preprocess(cv_image)

        self.stack_imgs.append(input_imgs)
        #print(len(self.stack_imgs)) ## will keep on discarding the previous frames and keep the latest four 

        if (len(self.stack_imgs) == 6):

            input = list(self.stack_imgs)
            input = torch.stack(input, dim=1)
            t = self.nstream(input)[0].detach().cpu().numpy()
        else:
            t = 1000  

		#t = self.predNet.getNearCollisionTime(cv_image)

        self.counter = self.counter + 1
        print(t)
        self.time_pub.publish(t)


def main(args):

    ic = image_converter()
    rospy.init_node('run_on_cam', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

