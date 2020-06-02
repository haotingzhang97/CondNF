import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim


def colorize(data, labels, p, d=1):
  # output shape: (num_of_images, 3, len1, len2)
  if d == 1:   # if data is of shape (num_of_images, 1, len1, len2)
      #trans_1to3d = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
      #data = trans_1to3d(data)
      data = np.float32(data)
      data = np.repeat(data,3,axis=1)
  else:
      data = np.float32(data)
  n = np.shape(data)[0]
  #l1 = np.shape(data)[1]; l2 = np.shape(data)[2]
  #new_data = np.ones((n, 3, l1, l2))
  new_data = np.ones(np.shape(data))
  data = data[:,0,:,:]
  for i in range(n):
    u = random.random()
    label = labels[i]
    if u < p[label,0]:
      new_data[i,1,:,:] = data[i,:,:]; new_data[i,2,:,:] = data[i,:,:]
      #new_data[i,1,:,:] = 1.0-data[i,:,:]; new_data[i,2,:,:] = 1.0-data[i,:,:]
    elif u < p[label,0] + p[label,1]:
      new_data[i,0,:,:] = data[i,:,:]; new_data[i,2,:,:] = data[i,:,:]
    else:
      new_data[i,0,:,:] = data[i,:,:]; new_data[i,1,:,:] = data[i,:,:]
  new_data = torch.tensor(new_data)
  return(new_data)


def colorize_red(data, d=1):
    if d == 1:   # if data is of shape (num_of_images, 1, len1, len2)
      #trans_1to3d = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
      #data = trans_1to3d(data)
      data = np.float32(data)
      data = np.repeat(data,3,axis=1)
    else:
      data = np.float32(data)
    n = np.shape(data)[0]
    new_data = np.ones(np.shape(data))
    data = data[:,0,:,:]
    for i in range(n):
        new_data[i,1,:,:] = data[i,:,:]; new_data[i,2,:,:] = data[i,:,:]
    new_data = torch.tensor(new_data)
    return(new_data)


def colorize_green(data, d=1):
    if d == 1:   # if data is of shape (num_of_images, 1, len1, len2)
      #trans_1to3d = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
      #data = trans_1to3d(data)
      data = np.float32(data)
      data = np.repeat(data,3,axis=1)
    else:
      data = np.float32(data)
    n = np.shape(data)[0]
    new_data = np.ones(np.shape(data))
    data = data[:,0,:,:]
    for i in range(n):
        new_data[i,0,:,:] = data[i,:,:]; new_data[i,2,:,:] = data[i,:,:]
    new_data = torch.tensor(new_data)
    return(new_data)


def colorize_blue(data, d=1):
    if d == 1:   # if data is of shape (num_of_images, 1, len1, len2)
      #trans_1to3d = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
      #data = trans_1to3d(data)
      data = np.float32(data)
      data = np.repeat(data,3,axis=1)
    else:
      data = np.float32(data)
    n = np.shape(data)[0]
    new_data = np.ones(np.shape(data))
    data = data[:,0,:,:]
    for i in range(n):
        new_data[i,0,:,:] = data[i,:,:]; new_data[i,1,:,:] = data[i,:,:]
    new_data = torch.tensor(new_data)
    return(new_data)