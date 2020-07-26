import numpy as np
import matplotlib.pyplot as plt
import torch
from colorization import *
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import copy

from colorization import *
from options.train_options import TrainOptions
from dataloader import load_data
from data import *
from models import create_model


p = np.zeros((10, 3))
for i in range(10):
    p[i, 0] = 0.1 + 0.7 * (9 - i) / 9
    p[i, 1] = 0.1
    p[i, 2] = 1 - p[i, 0] - p[i, 1]
root = './MNIST_data'
if not os.path.exists(root):
    os.mkdir(root)
trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.0,), (255.0,))])
# if not exist, download mnist dataset
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
test_set.data = 1.0 - test_set.data / 255.0
# transform from (n, 28, 28) to (n, 1, 32, 32)
test_set.data = torch.unsqueeze(test_set.data, 1)
transform_resize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
])
test_set_resized = torch.zeros([len(test_set.data), 1, 32, 32])
for i in range(len(test_set.data)):
    test_set_resized[i, :, :, :] = transform_resize(test_set.data[i, :, :, :])
#test_set_resized = test_set.data
test_data = test_set_resized
test_targets = test_set.targets
del test_set_resized, test_set

model = torch.load('/Users/Haoting/Desktop/MSc_code/savedmodels/model_cglow10.pt', map_location='cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_samples = 5

for digit in range(10):
    plt.figure()
    ind = np.where(test_targets == digit)[0]; i = ind[np.random.randint(0, len(ind), 1)]
    input = test_data[i, :, :, :].to(device).float()
    for nn in range(n_samples):
        y_sample,_ = model(input, reverse=True)
        y_sample = y_sample.detach().cpu().numpy()
        fig0 = np.swapaxes(np.swapaxes(np.squeeze(y_sample), 0, 1), 1, 2)
        ax = plt.subplot(1, n_samples, nn+1)
        ax.axis('off')
        ax.imshow(fig0)

'''
nll_mat = np.zeros((10,3))
for digit in range(10):
    ind = np.where(test_targets == digit)[0]; i = ind[np.random.randint(0, len(ind), 64)]
    input = test_data[i, :, :, :].to(device).float()
    cr = colorize_red(input)
    _, nll = model(input,cr.float())
    nll_mat[digit, 0] = np.mean(nll.detach().numpy())
    cg = colorize_green(input)
    _, nll = model(input,cg.float())
    nll_mat[digit, 1] = np.mean(nll.detach().numpy())
    cb = colorize_blue(input)
    _, nll = model(input,cb.float())
    nll_mat[digit, 2] = np.mean(nll.detach().numpy())
print(nll_mat)



opt = TrainOptions().parse()   # get training options
device = 'cuda' if torch.cuda.is_available() else 'cpu'
p = np.zeros((10, 3))
for i in range(10):
    p[i, 0] = 0.1 + 0.7 * (9 - i) / 9
    p[i, 1] = 0.1
    p[i, 2] = 1 - p[i, 0] - p[i, 1]
opt.p = p
# create a dataset given opt.dataset_mode and other options
train_data, train_targets, _, _, train_set_colorized = load_data(opt)
train_data = torch.repeat_interleave(train_data, 3, dim=1)
train_data = preprocess(train_data, 1.0, 0.0, opt.x_bins, True)
train_set_colorized = preprocess(train_set_colorized, 1.0, 0.0, opt.x_bins, True)
#train_set_colorized = preprocess(train_set_colorized, opt.label_scale, opt.label_bias, opt.y_bins, True)
plt.imshow(np.swapaxes(np.swapaxes(np.squeeze(train_data[26,:,:,:]), 0, 1), 1, 2))
plt.show()
plt.imshow(np.swapaxes(np.swapaxes(np.squeeze(train_set_colorized[26,:,:,:]), 0, 1), 1, 2))
plt.show()

#x = np.array([1.0,2.2,3.4])
#y = np.array([2.0,1.2,2.4])
#plt.plot(x,y)
#plt.show()




p = np.zeros((10, 3))
for i in range(10):
    p[i, 0] = 0.1 + 0.7 * (9 - i) / 9
    p[i, 1] = 0.1
    p[i, 2] = 1 - p[i, 0] - p[i, 1]
root = './MNIST_data'
if not os.path.exists(root):
    os.mkdir(root)
trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.0,), (255.0,))])
# if not exist, download mnist dataset
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
test_set.data = 1.0 - test_set.data / 255.0
# transform from (n, 28, 28) to (n, 1, 64, 64)
test_set.data = torch.unsqueeze(test_set.data, 1)
transform_resize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
])
test_set_resized = torch.zeros([len(test_set.data), 1, 64, 64])
for i in range(len(test_set.data)):
    test_set_resized[i, :, :, :] = transform_resize(test_set.data[i, :, :, :])
#test_set_resized = test_set.data
test_data = test_set_resized
test_targets = test_set.targets
del test_set_resized, test_set

model = torch.load('/Users/Haoting/Desktop/MSc_code/savedmodels/model_cglow01.pt', map_location=torch.device('cpu'))
#model = torch.load('/Users/Haoting/Desktop/MSc_code/savedmodels/model_cglow01.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_samples = 5
for digit in range(10):
    plt.figure()
    ind = np.where(test_targets == digit)[0]; i = ind[np.random.randint(0, len(ind), 1)]
    input = test_data[i, :, :, :].to(device).float()
    input = torch.repeat_interleave(input, 3, dim=1)
    for nn in range(n_samples):
        y_sample, _ = model(input, reverse=True)
        fig0 = np.swapaxes(np.swapaxes(np.squeeze(y_sample), 0, 1), 1, 2)
        ax = plt.subplot(1, n_samples, nn+1)
        ax.axis('off')
        ax.imshow(fig0)
'''