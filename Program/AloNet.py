"""
##########################################
AloNet Main Program
Author: Solam Lee, MD (solam@yonsei.ac.kr)
##########################################

< Information >

This program contains core modules for the model train and inference.
Please note that the images and the pixelwise annotations should be converted into numpy file using the "Numpy Converter" in order to use this program.

"""

# Import Library

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as utils
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn import init
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, jaccard_similarity_score, roc_curve, auc

%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
import PIL.Image as pilimg
import cv2
import albumentations as A

import time
import os
import copy
import random
from random import randint
import warnings
warnings.filterwarnings("ignore")



"""

< Settings >

"""

# Specify GPU number if you have multi GPUs.
# 0: Default
# -1: GPU parallelism

gpu_device_no = -1


# Where to save model parameters after training

save_path_mask_temporal_trained = "trained/mask_temporal.pt"
save_path_mask_midline_trained = "trained/mask_midline.pt"
save_path_target_trained = "trained/target.pt"

# Data path

use_internal_data = True

path_train_left_image_npy = "npy_by_direction/train/left_image.npy"
path_train_left_target_npy = "npy_by_direction/train/left_target.npy"
path_train_left_mask_npy = "npy_by_direction/train/left_mask.npy"

path_train_right_image_npy = "npy_by_direction/train/right_image.npy"
path_train_right_target_npy = "npy_by_direction/train/right_target.npy"
path_train_right_mask_npy = "npy_by_direction/train/right_mask.npy"

path_train_top_image_npy = "npy_by_direction/train/top_image.npy"
path_train_top_target_npy = "npy_by_direction/train/top_target.npy"
path_train_top_mask_npy = "npy_by_direction/train/top_mask.npy"

path_train_back_image_npy = "npy_by_direction/train/back_image.npy"
path_train_back_target_npy = "npy_by_direction/train/back_target.npy"
path_train_back_mask_npy = "npy_by_direction/train/back_mask.npy"

path_test_left_image_npy = "npy_by_direction/test/left_image.npy"
path_test_left_target_npy = "npy_by_direction/test/left_target.npy"
path_test_left_mask_npy = "npy_by_direction/test/left_mask.npy"

path_test_right_image_npy = "npy_by_direction/test/right_image.npy"
path_test_right_target_npy = "npy_by_direction/test/right_target.npy"
path_test_right_mask_npy = "npy_by_direction/test/right_mask.npy"

path_test_top_image_npy = "npy_by_direction/test/top_image.npy"
path_tset_top_target_npy = "npy_by_direction/test/top_target.npy"
path_test_top_mask_npy = "npy_by_direction/test/top_mask.npy"

path_test_back_image_npy = "npy_by_direction/test/back_image.npy"
path_test_back_target_npy = "npy_by_direction/test/back_target.npy"
path_test_back_mask_npy = "npy_by_direction/test/back_mask.npy"


# Train set split

test_size = 0.2
random_state = 2019

# Settings for train

train_for_mask_temporal = False
n_epoch_for_mask_temporal = 200
trainset_augmentation_for_mask_temporal = False
validset_augmentation_for_mask_temporal = False

train_for_mask_midline = False
n_epoch_for_mask_midline = 200
trainset_augmentation_for_mask_midline = True
validset_augmentation_for_mask_midline = False

train_for_target = False
n_epoch_for_target = 200
train_with_masking = True
masking_color = [ 0, 255, 255 ]
trainset_augmentation_for_target = True
validset_augmentation_for_target = False

model_auto_save = False
cumulative_save = False

# Batch size and class weight

batch_size = 24
class_weight = [1, 5]

# Settings for test with internal data

test_for_mask_temporal = True
test_for_mask_midline = True
test_for_target = True

test_img_show = True
max_show_image = 20



"""

< Functions >

"""


# Image viewer

def show_result (n_imgs, figsize, args):
    
    subplot = []
    plot = plt.figure (figsize=figsize)
    
    for i in range ( n_imgs ):

        subplot.append ( plot.add_subplot( 1, n_imgs, i+1 ) )
        
        img_title = args [i][0]
        img_type = args [i][1]
        img_data = args [i][2]
        
        if img_type == "image":
            subplot[i].imshow ( img_data )
        elif img_type == "binary":
            #subplot[i].imshow( img_data, cmap='plasma')
            subplot[i].imshow( img_data, vmin=0.0, vmax=1.0, cmap='seismic')

        subplot[i].set_title( img_title )
        subplot[i].axis('off')
        

# Tensor converter

def binarize(mask, thr=0, negative=False, two_layers=False):

    ones = torch.zeros(mask.shape)
    ones[mask > thr] = 1.0

    if negative:
        neg = torch.zeros(mask.shape)
        neg[mask >= thr] = 1.0
        return neg.squeeze(0)

    if two_layers:
        neg = torch.zeros(mask.shape)
        neg[mask < thr] = 1.0
        return torch.cat((neg, ones), dim=0)

    return ones.squeeze(0)


# Horizontal flip generator

def data_augmentation(image, mask, target, aug=True):

    if aug:
                
        image = np.array ( image )
        mask = np.array ( mask )
        target = np.array ( target )
        
        if randint ( 0, 100 ) < 50:        
            image = cv2.flip ( image, 1 )
            mask = cv2.flip ( mask, 1 )
            target = cv2.flip ( target, 1 )
            
    image = TF.to_tensor(image).float()
    mask = binarize(TF.to_tensor(mask)).long()
    target = binarize(TF.to_tensor(target)).long()
    
    return image, mask, target


# Dataset generator

class trainDataset(torch.utils.data.Dataset):
    def __init__(self, image, mask, target, augmentation=True):
        self.image = image.astype(np.uint8)
        self.mask = mask
        self.target = target
        self.augmentation = augmentation
        
    def __getitem__(self, index):
        get_image = self.image[index]
        get_mask = self.mask[index]
        get_target = self.target[index]
        return self.transform(get_image, get_mask, get_target)
    
    def transform(self, image, mask, target):
        image, mask, target = data_augmentation(image, mask, target, self.augmentation)
        return image, mask, target
    
    def __len__(self):
        return len(self.image)


# U-Net-based Network Configuration

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )   

class UNet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.dconv_down5 = double_conv(512, 1024)

        self.dropout = nn.Dropout(p=0.25)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up4 = double_conv(512 + 1024, 512)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, 2, 1)
        
    def forward(self, x, phase='valid'):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)   
        
        x = self.dconv_down5(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv4], dim=1)
        if phase=='train':
            x = self.dropout(x)

        x = self.dconv_up4(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        if phase=='train':
            x = self.dropout(x)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       
        if phase=='train':
            x = self.dropout(x)

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        if phase=='train':
            x = self.dropout(x)
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out


# Loss function

def cross_entropy2d(input, target, weight=None, size_average=True):
    
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    if h != ht and w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
        
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    if scale_weight is None:
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            target.device
        )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


# Loader Generator

def fit(epoch,model,data_loader,phase='valid',mode='target',volatile=False):

    global criterion, optimizer, exp_lr_scheduler
    
    if phase == 'train':
        exp_lr_scheduler.step()
        model.train()
    if phase == 'valid':
        model.eval()
    running_loss = 0.0
    
    for batch_idx , (data, mask, target) in enumerate(data_loader):

        
        if mode == "target":
            inputs,target = data.cpu(),target.cpu()
            if is_cuda:
                inputs,target = data.cuda(),target.cuda()
        
        elif mode == "mask":
            inputs,target = data.cpu(),mask.cpu()
            if is_cuda:
                inputs,target = data.cuda(),mask.cuda()
            
            
        inputs , target = Variable(inputs),Variable(target)

        if phase == 'train':
            optimizer.zero_grad()

        output = model(inputs, phase)
        wt = torch.tensor(class_weight).cuda()
        loss = multi_scale_cross_entropy2d(output,target, weight=wt)
        
        running_loss += loss.data.item()
        
        if phase == 'train':
            loss.backward()
            optimizer.step()
    
    loss = running_loss/len(data_loader.dataset)
    return loss



# Train Module

def AloNet_train ( title, train_switch, image_set, mask_set, target_set, mode, model_save_path, n_epoch, train_aug, valid_aug  ):
    
    if train_switch == True:

        global criterion, optimizer, exp_lr_scheduler
        
        model = UNet()
        if is_cuda:
            if gpu_device_no == -1:
                model = nn.DataParallel(model)
            else:
                torch.cuda.set_device(gpu_device_no)
            model = model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=0.0005)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        if mode == "target" and train_with_masking == True:
            image_set[ mask_set == 0 ] = masking_color
            target_set[ mask_set == 0 ] = 0
        
        train_image, valid_image, train_mask, valid_mask, train_target, valid_target = train_test_split(image_set, mask_set, target_set, test_size=test_size, random_state=random_state)

        traindataset = trainDataset(train_image, train_mask, train_target, augmentation=train_aug )
        validdataset = trainDataset(valid_image, valid_mask, valid_target, augmentation=valid_aug )

        trainloader = torch.utils.data.DataLoader(traindataset, batch_size = batch_size, shuffle=True, num_workers=0, pin_memory=False)
        validloader = torch.utils.data.DataLoader(validdataset, batch_size = batch_size, shuffle=True, num_workers=0, pin_memory=False)

        print ( "< Train for " + title + " >" )

        since = time.time()

        loss_min = 1

        for epoch in range( n_epoch ):
            epoch_loss = fit(epoch,model,trainloader,phase='train',mode=mode)
            val_epoch_loss = fit(epoch,model,validloader,phase='valid',mode=mode)
            time_elapsed = time.time() - since

            print('=== Epoch {} (Total elapsed: {:.0f}m {:.0f}s) ===\nTrain Loss: {:.5f}\nValid Loss: {:.5f}'.format(epoch+1, time_elapsed // 60, time_elapsed % 60, epoch_loss, val_epoch_loss))        

            if model_auto_save == True and loss_min > val_epoch_loss:
                loss_min = val_epoch_loss
                if cumulative_save == True:
                    torch.save(model.module.state_dict(),model_save_path+".epoch."+str(epoch+1))
                    torch.save(model.module.state_dict(),model_save_path)
                else:
                    torch.save(model.module.state_dict(),model_save_path)

                print ("< Model Saved >")

        time_elapsed = time.time() - since
        print()
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


# Test Module

def AloNet_test ( title, test_switch, image_set, mask_set, target_set, mode, model_mask_path, model_target_path ):
    
    model = UNet()
    
    if is_cuda:
        torch.cuda.set_device(3)
        model = model.cuda()        

    valid_mask, train_target, valid_target = image_set, mask_set, target_set

    validdataset = trainDataset(valid_image, valid_mask, valid_target, augmentation=False )
    validloader = torch.utils.data.DataLoader(validdataset, batch_size = 1, shuffle=False, num_workers=0, pin_memory=False)

    n_data = len(validloader.dataset)
    mask_pred_col = np.empty((n_data,height,width))
    target_pred_col = np.empty((n_data,height,width))
    mask_col = np.empty((n_data,height,width))
    target_col = np.empty((n_data,height,width))
    
    with torch.no_grad():
        for batch_idx, (inputs,mask,target) in enumerate(validloader):
            
            if is_cuda:
                inputs,mask,target = inputs.cuda(),mask.cuda(),target.cuda()
                
            inputs, mask, target = Variable(inputs), Variable(mask), Variable(target)

            model.load_state_dict(torch.load(model_mask_path))
            model.eval()        
            output_mask = model(inputs).cpu()    
            
            inputs_mask = np.array ( np.transpose (inputs.cpu(), (0,2,3,1) ) )
            target_mask = np.array ( target.cpu() )
            
            mask_for_inputs = np.array ( np.transpose (output_mask.cpu(), (1,0,2,3) ) )
            
            inputs_mask [ mask_for_inputs[1] < 0.5 ] = [ masking_color[0] / 255, masking_color[1] / 255, masking_color[2] / 255 ]
            target_mask [ mask_for_inputs[1] < 0.5 ] = 0
                    
            inputs_mask = torch.from_numpy( np.transpose ( inputs_mask, (0, 3, 1, 2 ) ) )
            inputs_mask = Variable ( inputs_mask.cuda() )
            
            model.load_state_dict(torch.load(model_target_path))
            model.eval()        
            output_target = model(inputs).cpu()
            output_target_mask = model(inputs_mask).cpu()

            inputs = inputs.cpu()
            inputs_mask = inputs_mask.cpu()
            mask = mask.cpu()
            target = target.cpu()     

            input_image = np.array ( np.transpose(inputs[0],(1,2,0)) )
            input_mask_image = np.array ( np.transpose(inputs_mask[0],(1,2,0)) )
            output_mask = np.array ( torch.sigmoid ( output_mask[0][1] ) )
            output_target = np.array ( torch.sigmoid ( output_target[0][1] ) )
            output_target_mask = np.array ( torch.sigmoid ( output_target_mask[0][1] ) )
        
            output_synthesize = np.copy ( output_target )
            output_synthesize [ output_mask < 0.6 ] = 0 

            if batch_idx < max_show_image:
            
            
                anony = cv2.blur ( input_image, (40,40) )
                                
                show_result ( n_imgs = 3,
                             figsize = (16, 8),
                             args = [
                                [ "Image", "image", input_image  ],
                                [ "Mask prediction", "binary", output_mask ],
                                [ "Target after mask", "binary", output_target_mask ]
                             ])
                
            mask_pred_col[batch_idx]=output_mask
            mask_col[batch_idx]=mask
            target_pred_col[batch_idx]=output_target_mask
            target_col[batch_idx]=target_mask




"""

< Main Program >

"""

# Check CUDA availability

if torch.cuda.is_available():
    is_cuda = True
else:
    is_cuda = False


# Load Train Data

left_image_npy = np.load(path_train_left_image_npy)
left_target_npy = np.load(path_train_left_target_npy)
left_mask_npy = np.load(path_train_left_mask_npy)

right_image_npy = np.load(path_train_right_image_npy)
right_target_npy = np.load(path_train_right_target_npy)
right_mask_npy = np.load(path_train_right_mask_npy)

top_image_npy = np.load(path_train_top_image_npy)
top_target_npy = np.load(path_train_top_target_npy)
top_mask_npy = np.load(path_train_top_mask_npy)

back_image_npy = np.load(path_train_back_image_npy)
back_target_npy = np.load(path_train_back_target_npy)
back_mask_npy = np.load(path_train_back_mask_npy)

for i in range ( len ( right_image_npy ) ):
    right_image_npy[i] = cv2.flip ( right_image_npy[i], 1 )
    right_mask_npy[i] = cv2.flip ( right_mask_npy[i], 1 )
    right_target_npy[i] = cv2.flip ( right_target_npy[i], 1 )

temporal_image_npy = np.concatenate ( ( left_image_npy, right_image_npy ) )
temporal_mask_npy = np.concatenate ( ( left_mask_npy, right_mask_npy ) )
temporal_target_npy = np.concatenate ( ( left_target_npy, right_target_npy ) )

midline_image_npy = np.concatenate ( ( top_image_npy, back_image_npy ) )
midline_target_npy = np.concatenate ( ( top_target_npy, back_target_npy ) )
midline_mask_npy = np.concatenate ( ( top_mask_npy, back_mask_npy ) )

image_npy = np.concatenate ( ( left_image_npy, right_image_npy, top_image_npy, back_image_npy ) )
mask_npy = np.concatenate ( ( left_mask_npy, right_mask_npy, top_mask_npy, back_mask_npy ) )
target_npy = np.concatenate ( ( left_target_npy, right_target_npy, top_target_npy, back_target_npy ) )


# Train for Temporal Subset of Scalp Identifier

AloNet_train ( title = "mask_temporal",
              train_switch = train_for_mask_temporal,
              image_set = temporal_image_npy,
              mask_set = temporal_mask_npy,
              target_set = temporal_target_npy,
              mode = "mask",
              model_save_path = save_path_mask_temporal_trained,
              n_epoch = n_epoch_for_mask_temporal,
              train_aug = trainset_augmentation_for_mask_temporal,
              valid_aug = validset_augmentation_for_mask_temporal)


# Train for Midline Subset of Scalp Identifier

AloNet_train ( title = "mask_midline",
               train_switch = train_for_mask_midline,
               image_set = midline_image_npy,
               mask_set = midline_mask_npy,
               target_set = midline_target_npy,
               mode = "mask",
               model_save_path = save_path_mask_midline_trained,
               n_epoch = n_epoch_for_mask_midline,
               train_aug = trainset_augmentation_for_mask_midline,
               valid_aug = validset_augmentation_for_mask_midline)


# Train for Hair Loss Identifier

AloNet_train ( title = "target",
               train_switch = train_for_target,
               image_set = image_npy,
               mask_set = mask_npy,
               target_set = target_npy,
               mode = "target",
               model_save_path = save_path_target_trained,
               n_epoch = n_epoch_for_target,
               train_aug = trainset_augmentation_for_target,
               valid_aug = validset_augmentation_for_target)


# Change Dataset into Test Data

left_image_npy = np.load(path_test_left_image_npy)
left_target_npy = np.load(path_test_left_target_npy)
left_mask_npy = np.load(path_test_left_mask_npy)

right_image_npy = np.load(path_test_right_image_npy)
right_target_npy = np.load(path_test_right_target_npy)
right_mask_npy = np.load(path_test_right_mask_npy)

top_image_npy = np.load(path_test_top_image_npy)
top_target_npy = np.load(path_test_top_target_npy)
top_mask_npy = np.load(path_test_top_mask_npy)

back_image_npy = np.load(path_test_back_image_npy)
back_target_npy = np.load(path_test_back_target_npy)
back_mask_npy = np.load(path_test_back_mask_npy)

for i in range ( len ( right_image_npy ) ):
    right_image_npy[i] = cv2.flip ( right_image_npy[i], 1 )
    right_mask_npy[i] = cv2.flip ( right_mask_npy[i], 1 )
    right_target_npy[i] = cv2.flip ( right_target_npy[i], 1 )

temporal_image_npy = np.concatenate ( ( left_image_npy, right_image_npy ) )
temporal_mask_npy = np.concatenate ( ( left_mask_npy, right_mask_npy ) )
temporal_target_npy = np.concatenate ( ( left_target_npy, right_target_npy ) )

midline_image_npy = np.concatenate ( ( top_image_npy, back_image_npy ) )
midline_target_npy = np.concatenate ( ( top_target_npy, back_target_npy ) )
midline_mask_npy = np.concatenate ( ( top_mask_npy, back_mask_npy ) )

image_npy = np.concatenate ( ( left_image_npy, right_image_npy, top_image_npy, back_image_npy ) )
mask_npy = np.concatenate ( ( left_mask_npy, right_mask_npy, top_mask_npy, back_mask_npy ) )
target_npy = np.concatenate ( ( left_target_npy, right_target_npy, top_target_npy, back_target_npy ) )


# Test for Temporal Subset

AloNet_test ( title = "test",
             test_switch = test_for_target,
             image_set = temporal_image_npy,
             mask_set = temporal_mask_npy,
             target_set = temporal_target_npy,
             mode = "target",
             model_mask_path = save_path_mask_midline_trained,
             model_target_path = save_path_target_trained )


# Test for Midline Subset

AloNet_test ( title = "test",
             test_switch = test_for_target,
             image_set = midline_image_npy,
             mask_set = midline_mask_npy,
             target_set = midline_target_npy,
             mode = "target",
             model_mask_path = save_path_mask_midline_trained,
             model_target_path = save_path_target_trained )

