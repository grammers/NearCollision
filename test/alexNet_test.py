#!/usr/bin/env python
import image_prosesser
from collections import deque 

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

import torch
import torchvision.models as models
import torch.nn as nn 

path = '/home/grammers/catkin_ws/src/nearCollision/data/'
#path = '/home/hexa/catkin_workspaces/catkin_samuel/src/nearCollision/data/'
#mean = 2.26691915033756 

class net_loader:

    def __init__(self):
        self.model = models.alexnet(pretrained=True)
        self.num_final_in = self.model.classifier[-1].in_features
        self.NUM_CLASSES = 1000
        self.model.classifier[-1] = nn.Linear(
            self.num_final_in, self.NUM_CLASSES) ## Regressed output
    #    self.model = nn.Linear(self.num_final_in, NUM_CLASSES)
        # load the modals
        self.load_weights(path + 'trained_models/alexNet/alexnet.pth')
        self.load_pred()
        self.load_weights(path + 'trained_models/alexNet/alexNet_6s002')
        
        self.model = self.model.cuda()
        self.model.eval()
    
    def load_pred(self):
        self.model.classifier[-1] = nn.Linear(
        self.num_final_in, self.NUM_CLASSES)
        num_features = self.model.classifier[6].in_features
        features = list(self.model.classifier.children())[:-3]
        
        features.extend([
            nn.Linear(num_features,2048),
            nn.ReLU(), nn.Linear(2048,2)
        ])
        self.model.classifier = nn.Sequential(*features)

    def load_weights(self, model_path):
        self.model.load_state_dict(
            torch.load(model_path))

    def feed_net(self, stack_img):
        return self.model(stack_img.pop())[0].data
        
        
class ROS_runner:
    
    def __init__(self):
        self.image_prosesser = image_prosesser.Image_prosesser(1)
        self.network = net_loader()
        
        #ros setup
        self.image_sub = rospy.Subscriber(
        #"/usb_cam/image_raw", Image, self.callback)
        "/image_slow", Image, self.callback)
        
        # publiched topics
        self.time_pub_left = rospy.Publisher(
        'near_collision_time/left', Float32, queue_size = 10)
        self.time_pub_right = rospy.Publisher(
        'near_collision_time/right', Float32, queue_size = 10)

        self.i = 0
        
    def callback(self, data):
        self.image_prosesser.preprocess(data) 
       
        # if is used if a mulit images steam is used
        # images feed thrue the network
        if (self.image_prosesser.is_filed()):
            t = self.network.feed_net(self.image_prosesser.stack())
        else:
            t = 1000

        self.time_pub_left.publish(t[0])
        self.time_pub_right.publish(t[1])
        self.i += 1
        print(self.i)
        print(t)

def main():
    ros_runner = ROS_runner()
    rospy.init_node('run_on_cam', anonymous=True)
    print('setup complet')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllwindows()

if __name__ == '__main__':
    main()
