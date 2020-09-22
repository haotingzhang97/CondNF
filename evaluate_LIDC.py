import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data.sampler import SubsetRandomSampler
import copy

from colorization import *
from options.train_options import TrainOptions
from data.load_LIDC_data import LIDC_IDRI
from data import *
from models import create_model
from models.utils import *

model = torch.load('/Users/haoting/Desktop/MSc_code/savedmodel/model_lidc_0916_2.pt', map_location=torch.device('cpu'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = LIDC_IDRI(dataset_location='LIDCdata/')

test_indices = np.load('/Users/haoting/Desktop/MSc_code/savedmodel/test_indices.npy')
test_indices = test_indices[0:5]
test_sampler = SubsetRandomSampler(test_indices)
test_loader = Data.DataLoader(dataset, batch_size=1, sampler=test_sampler, shuffle=False)
print("Number of test patches:", len(test_indices))
L = len(test_indices)

print('Start evaluation')
ged1 = np.zeros((L)); ged2 = np.zeros((L)); ged3 = np.zeros((L))
j = 0
y_mat1 = np.zeros((L,1,128,128)); y_mat2 = np.zeros((L,1,128,128))
with torch.no_grad():
    for i, (x, y, _) in enumerate(test_loader):
        x = x.to(device).float()
        # y = torch.unsqueeze(y, 1)
        y = y.to(device).float()
        y0, _ = model.forward(x, reverse=True)
        y1, _ = model.forward(x, reverse=True)
        y0 = y0[0,1,:,:]
        y1 = y1[0,1,:,:]
        y = torch.round(y).detach().cpu().numpy()
        y_mat1[np.array([j]), :, :, :] = y
        y = y[0,0,:,:]
        y0 = torch.round(y0).detach().cpu().numpy()
        y1 = torch.round(y1).detach().cpu().numpy()
        intersection = np.logical_and(y, y0)
        union = np.logical_or(y, y0)
        if np.sum(union) > 0:
            ged1[j] = 1 - np.sum(intersection) / np.sum(union)
        else:
            ged1[j] = 0
        intersection = np.logical_and(y0, y1)
        union = np.logical_or(y0, y1)
        if np.sum(union) > 0:
            ged3[j] = 1 - np.sum(intersection) / np.sum(union)
        else:
            ged3[j] = 0
        j += 1
        if j == 500:
            print("progress: 500/3019")
        if j == 1000:
            print("progress: 1000/3019")
        if j == 2000:
            print("progress: 2000/3019")
j = 0
with torch.no_grad():
    for i, (x, y, _) in enumerate(test_loader):
        y_mat2[np.array([j]), :, :, :] = y
        j += 1
for i in range(L):
    y0 = y_mat1[i, :, :, :]; y1 = y_mat2[i, :, :, :]
    intersection = np.logical_and(y0, y1)
    union = np.logical_or(y0, y1)
    if np.sum(union) > 0:
        ged2[i] = 1 - np.sum(intersection) / np.sum(union)
    else:
        ged2[i] = 0
ged_square = 2 * np.mean(ged1) - np.mean(ged2) - np.mean(ged3)
print(ged_square)
print('Evaluation finished')
