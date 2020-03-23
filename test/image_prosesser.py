#!/usr/bin/env python
import cv2

from collections import deque
from torchvision.transforms import transforms
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image


class Image_prosesser:
    
    def __init__(self, stream_len):
        self.buf_size = stream_len
        self.bridge = CvBridge()
        self.stack_images = deque(maxlen = self.buf_size)

    def preprocess(self, image):
        # resice to desierd size
        cv_img = self.bridge.imgmsg_to_cv2(image, "bgr8")
        cv_img = cv2.resize(cv_img, (224, 224))

        #normalize the image not chor way
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
        
        transform = transforms.Compose(
        [transforms.ToTensor(), normalize])
        
        cv_img = transform(cv_img)

        ## unsqueeze it to become torch.Size([1, 3, 224, 224])
        cv_img = cv_img.unsqueeze(0)
        cv_img = cv_img.float().cuda()
        
        self.stack_images.append(cv_img)

    def is_filed(self):
        return len(self.stack_images) == self.buf_size

    def stack(self):
        return self.stack_images
        
