#!/usr/bin/python
# coding: utf-8

# In[1]:


import torch
import torchvision.models as models
import h5py 
from logger import Logger
from torchvision.transforms import transforms 
import torch.utils.data as data
import numpy as np 
import pdb
#import matplotlib.pyplot as plt
import torch.nn as nn 
import torch.optim as optim 
from torch.autograd import Variable
import shutil
import os 
import random
import torch.nn.functional as F
import math 


# In[2]:
path = '/home/grammers/catkin_ws/src/nearCollision/data/'

# extraxts data form .h5 flie
class FrameDataset(data.Dataset):
    
    def __init__(self, f, transform=None, test = False):
        self.f = f 
        self.transform = transform 
        self.test = test
        
    def __getitem__(self, index):

        rgb = np.array(self.f["image"][index])
        label = np.array(self.f["lable"][index]) 
        
        t_rgb = torch.zeros(3, 224, 224)
        
        prob = random.uniform(0, 1)
        prob2 = random.uniform(0, 1)

        if self.transform is not None:
            if (prob > 0.5 and not self.test):
                flip_transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(1.0)])
                rgb[:,:,:] = flip_transform(rgb[:,:,:])
            if (prob2 > 0.5 and not self.test):
                color_jitter_transform = transforms.Compose([transforms.ToPILImage() ,transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.2)])
                rgb[:,:,:] = color_jitter_transform(rgb[:,:,:])

            t_rgb[:,:,:] = self.transform(rgb[:,:,:])                
        
        return t_rgb, label
    
    def __len__(self):
        return len(self.f["image"])


# In[3]:


def load_model_weights(MODEL_PATH):
    checkpoint_dict = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint_dict)
    
def save_model_weights(epoch_num):
    model_file = path + 'trained_models/resNet/resNet_6s' + str(epoch_num).zfill(3)
    torch.save(model.state_dict(), model_file)


# In[4]:


model = models.resnet101(pretrained=True)
#num_final_in = model.classifier[-1].in_features
NUM_CLASSES = 1000
#model.classifier[-1] = nn.Linear(num_final_in, NUM_CLASSES)

model_path = path + 'trained_models/' + 'voc_2007_trainval/res101_faster_rcnn_iter_70000.pkl'
load_model_weights(model_path)

print(model)
model.classifier[-1] = nn.Linear(num_final_in, 1) ## Regressed output
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-3] # Remove last layer
# modification to network ockur here
features.extend([nn.Linear(num_features, 2048), nn.ReLU(), nn.Linear(2048, 2)]) # Add our layer with 3 outputs
model.classifier = nn.Sequential(*features) # Replace the model classifier

print(model)
'''
# ust to restart att custom point
epoch_num = 53
MODEL_PATH = path + 'trained_models/alexNet/alexNet_6s' + str(epoch_num).zfill(3)
load_model_weights(MODEL_PATH)
# In[5]:
'''

if os.path.exists(path + 'trained_models/resNet/resNet6s'):
    shutil.rmtree(path + 'trained_models/resNet/resNet6s')
logger = Logger('resNet6s', name='performance_curves')


# In[6]:


model = model.cuda()
test_dir = path + 'h5_file/duble_human_test/'
train_dir = path + 'h5_file/duble_human_train/'
#hfp_test = h5py.File('/.h5', 'r')
#hfp_train = h5py.File('/mnt/hdd1/aashi/cmu_data/SingleImageTrain.h5', 'r')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#optimizer = optim.SGD(model.parameters(),0.0005)
optimizer = optim.SGD(model.parameters(),0.0001)
params = model.parameters()

## Test data: 949; Train data: 12146; Mean: 2.909sec; Variance: 2.2669sec

offset = 0
iterations = offset
epochs = 100 - offset
criterion = nn.MSELoss()

best_err = 100

for e in range(epochs):
    print(e + offset)
    model.eval()
    mean_err = 0.0
    # read ewry file in the test dir
    for test_file in os.listdir(test_dir):
        hfp_test = h5py.File(test_dir + test_file, 'r') 
        #print(test_file)
        test_loader = data.DataLoader(FrameDataset(f = hfp_test, transform = transforms.Compose([transforms.ToTensor(), normalize]), test = True), 
                                   batch_size=1)
        err = 0.0
        for iter, (rgb, label) in enumerate(test_loader, 0):
            rgb = rgb.float().cuda()
            label = Variable(label.float().cuda())
            label = label.unsqueeze(-1)
            outputs = model(rgb)
            # ther are thre outputs to consider
            for i in range(0, 2):
                #add on error of eatch image
                err += abs(outputs[0][i].data.cpu().numpy() - label[0][i].data.cpu().numpy())
        logger.scalar_summary('Test Error', err/len(test_loader), e + offset)
        mean_err += err/len(test_loader)
        #prints the mean error
        #print('test err = %s' % str(err/len(test_loader)))
        hfp_test.close()
    # mean of the entire test setion
    tot_men_err = mean_err/26.0
    print('mean err = %s' % str(tot_men_err))
    
    f = open(path + 'trained_models/resNet/loog.txt', 'a')
    f.write(str(e-1 + offset)  + ' mean err = ' + str(tot_men_err) + '\n')
    f.close()

    model.train() 
    # train on all data in training dir
    for train_file in os.listdir(train_dir):
        hfp_train = h5py.File(train_dir + train_file, 'r') 
        #print(train_file)
        # batch_size shoud myght be somthing ells but it don't work with out modifications
        batch_size = 20
        train_loader = data.DataLoader(
            FrameDataset(f = hfp_train, transform = transforms.Compose(
            [transforms.ToTensor(), normalize]),
            test = False),
            batch_size=batch_size, shuffle=True)

        for iter, (rgb, label) in enumerate(train_loader, 0):
            rgb = Variable(rgb.float().cuda())
            label = Variable(label.float().cuda())
            optimizer.zero_grad()
            #label = label.unsqueeze(-1)
            outputs = model(rgb)
            #print(outputs)
            #torch.nn.utils.clip_grad_norm_(params, 10)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            iterations += 1
        logger.scalar_summary('training_loss', loss.data.cpu().numpy(), iterations)
        hfp_train.close()
        #print('trained')

    best_err = tot_men_err
    save_model_weights(e + offset)
    print('saved')

# In[ ]:


#hfp_test.close()
#hfp_train.close()

