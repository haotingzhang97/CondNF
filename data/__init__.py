import torch.utils.data
import os
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from colorization import *


def load_data(opt):
    dataset_name = opt.dataset_name
    if dataset_name == 'MNIST':
        root = './MNIST_data'
        if not os.path.exists(root):
            os.mkdir(root)
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.0,), (255.0,))])
        # if not exist, download mnist dataset
        train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
        test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
        train_set.data = 1.0 - train_set.data / 255.0
        test_set.data = 1.0 - test_set.data / 255.0
        # comment or uncomment the next 6 lines (or previous 2 lines) depending on version of torchvision
        #train_set.data = 1.0 - train_set.train_data / 255.0
        #test_set.data = 1.0 - test_set.test_data / 255.0
        #del train_set.train_data
        #del test_set.test_data
        #train_set.targets = train_set.train_labels
        #test_set.targets = test_set.test_labels
        N = len(train_set.targets)
        K = np.floor(N*opt.subset)
        # Randomly select a subset of training data (optional)
        indices = list(range(N))
        np.random.seed(123)
        np.random.shuffle(indices)
        train_sub = indices[0:int(K)]
        train_set.data = train_set.data[train_sub]
        train_set.targets = train_set.targets[train_sub]
        # transform from (n, 28, 28) to (n, 1, opt.newsize, opt.newsize)
        train_set.data = torch.unsqueeze(train_set.data, 1);
        test_set.data = torch.unsqueeze(test_set.data, 1)
        if opt.resize == True:
            transform_resize = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((opt.newsize, opt.newsize)),
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
            ])
            train_set_resized = torch.zeros([len(train_set.data), 1, opt.newsize, opt.newsize])
            test_set_resized = torch.zeros([len(test_set.data), 1, opt.newsize, opt.newsize])
            for i in range(len(train_set.data)):
                train_set_resized[i, :, :, :] = transform_resize(train_set.data[i, :, :, :])
            for i in range(len(test_set.data)):
                test_set_resized[i, :, :, :] = transform_resize(test_set.data[i, :, :, :])
            train_set.data = train_set_resized
            test_set.data = test_set_resized
            del train_set_resized, test_set_resized
        train_set_colorized = colorize(train_set.data, train_set.targets, opt.p, 1)
        return train_set.data, train_set.targets, test_set.data, test_set.targets, train_set_colorized


def load_data_seg(opt):
    dataset_name = opt.dataset_name
    if dataset_name == 'MNIST':
        root = './MNIST_data'
        if not os.path.exists(root):
            os.mkdir(root)
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.0,), (255.0,))])
        # if not exist, download mnist dataset
        train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
        test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
        train_set.data = 1.0 - train_set.data / 255.0
        test_set.data = 1.0 - test_set.data / 255.0
        # comment or uncomment the next 6 lines (or previous 2 lines) depending on version of torchvision
        #train_set.data = 1.0 - train_set.train_data / 255.0
        #test_set.data = 1.0 - test_set.test_data / 255.0
        #del train_set.train_data
        #del test_set.test_data
        #train_set.targets = train_set.train_labels
        #test_set.targets = test_set.test_labels
        N = len(train_set.targets)
        K = np.floor(N*opt.subset)
        # Randomly select a subset of training data (optional)
        indices = list(range(N))
        np.random.seed(123)
        np.random.shuffle(indices)
        train_sub = indices[0:int(K)]
        train_set.data = train_set.data[train_sub]
        train_set.targets = train_set.targets[train_sub]
        # transform from (n, 28, 28) to (n, 1, opt.newsize, opt.newsize)
        train_set.data = torch.unsqueeze(train_set.data, 1);
        test_set.data = torch.unsqueeze(test_set.data, 1)
        if opt.resize == True:
            transform_resize = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((opt.newsize, opt.newsize)),
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
            ])
            train_set_resized = torch.zeros([len(train_set.data), 1, opt.newsize, opt.newsize])
            test_set_resized = torch.zeros([len(test_set.data), 1, opt.newsize, opt.newsize])
            for i in range(len(train_set.data)):
                train_set_resized[i, :, :, :] = transform_resize(train_set.data[i, :, :, :])
            for i in range(len(test_set.data)):
                test_set_resized[i, :, :, :] = transform_resize(test_set.data[i, :, :, :])
            train_set.data = train_set_resized
            test_set.data = test_set_resized
            del train_set_resized, test_set_resized
        train_set_colorized = colorize(train_set.data, train_set.targets, opt.p, 1)
        train_set_seg = torch.tensor(np.zeros((int(K), 4, opt.newsize, opt.newsize)))
        for i in range(int(K)):
            train_set_colorized_i = np.squeeze(train_set_colorized[i, :, :, :].detach().numpy())
            rmat = train_set_colorized_i[0, :, :]
            gmat = train_set_colorized_i[1, :, :]
            bmat = train_set_colorized_i[2, :, :]
            for k1 in range(opt.newsize):
                for k2 in range(opt.newsize):
                    if gmat[k1, k2] < 0.9 and bmat[k1, k2] < 0.9:
                        train_set_seg[i, 1, k1, k2] = 1
                    elif rmat[k1, k2] < 0.9 and bmat[k1, k2] < 0.9:
                        train_set_seg[i, 2, k1, k2] = 1
                    elif rmat[k1, k2] < 0.9 and gmat[k1, k2] < 0.9:
                        train_set_seg[i, 3, k1, k2] = 1
                    else:
                        train_set_seg[i, 0, k1, k2] = 1
        return train_set.data, train_set.targets, test_set.data, test_set.targets, train_set_colorized, train_set_seg


def preprocess(x, scale, bias, bins, noise=False):

    x = x / scale
    x = x - bias

    if noise == True:
        if bins == 2:
            x = x + torch.zeros_like(x).uniform_(-0.5, 0.5)
        else:
            #x = x + torch.zeros_like(x).uniform_(0, 1/bins)
            x = x + torch.zeros_like(x).uniform_(-1 / bins, 1 / bins)
    return x


def postprocess(x, scale, bias):

    x = x + bias
    x = x * scale
    return x


def convert_to_img(y):
    import skimage.color
    import skimage.util
    import skimage.io

    C = y.size(1)

    transform = transforms.ToTensor()
    colors = np.array([[0,0,0],[255,255,255]])/255

    if C == 1:
        seg = torch.squeeze(y, dim=1).cpu().numpy()
        seg = np.nan_to_num(seg)
        seg = np.clip(np.round(seg),a_min=0, a_max=1)

    if C > 1:
        seg = torch.mean(y, dim=1, keepdim=False).cpu().numpy()
        seg = np.nan_to_num(seg)
        seg = np.clip(np.round(seg),a_min=0, a_max=1)

    B,C,H,W = y.size()
    imgs = list()
    for i in range(B):
        label_i = skimage.color.label2rgb(seg[i], colors=colors)
        label_i = skimage.util.img_as_ubyte(label_i)
        imgs.append(transform(label_i))
    return imgs, seg
