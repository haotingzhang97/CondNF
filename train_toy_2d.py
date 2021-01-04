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
import copy

from colorization import *
from options.train_options import TrainOptions
from data import *
from models import create_model


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p = np.zeros((10, 3))
    for i in range(10):
        p[i, 0] = 0.1 + 0.7 * (9 - i) / 9
        p[i, 1] = 0.1
        p[i, 2] = 1 - p[i, 0] - p[i, 1]
    opt.p = p

    if opt.model_name == 'cglow':
        opt.x_size = (opt.input_nc, opt.newsize, opt.newsize)
        opt.y_size = (opt.output_nc, opt.newsize, opt.newsize)

    if opt.sample_method == 0:
        # create a dataset given opt.dataset_mode and other options
        train_data, train_targets, _, _, _, _ = load_data_seg(opt)
        full_size = len(train_targets)
        indices = list(range(full_size))
        split = int(np.floor(opt.val_proportion * full_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        val_data = train_data[val_indices, :, :, :]
        val_targets = train_targets[val_indices]
        #val_set_seg = train_set_seg[val_indices, :, :, :]
        train_data = train_data[train_indices, :, :, :]
        train_targets = train_targets[train_indices]
        #train_set_seg = train_set_seg[train_indices, :, :, :]

        ind = np.union1d(np.where(train_targets.detach().cpu().numpy() == 0)[0],
                         np.where(train_targets.detach().cpu().numpy() == 8)[0])
        train_data0 = train_data[ind]
        train_targets0 = train_targets[ind]
        train_set_seg0 = torch.zeros((len(train_targets0), 2, opt.newsize, opt.newsize))
        for i in range(len(train_targets0)):
            u = np.random.uniform(0.0, 1.0)
            if train_targets0[i] == 0:
                if u < 0.8:
                    train_set_seg0[i, 0, :, :] = torch.ones_like(train_data0[i, 0, :, :]) - torch.round(train_data0[i, 0, :, :])
                else:
                    train_set_seg0[i, 1, :, :] = torch.ones_like(train_data0[i, 0, :, :]) - torch.round(train_data0[i, 0, :, :])
            if train_targets0[i] == 8:
                if u < 0.8:
                    train_set_seg0[i, 1, :, :] = torch.ones_like(train_data0[i, 0, :, :]) - torch.round(train_data0[i, 0, :, :])
                else:
                    train_set_seg0[i, 0, :, :] = torch.ones_like(train_data0[i, 0, :, :]) - torch.round(train_data0[i, 0, :, :])

        dataset_size = len(train_targets0)  # get the number of images in the dataset.

        ind = np.union1d(np.where(val_targets.detach().cpu().numpy() == 0)[0],
                         np.where(val_targets.detach().cpu().numpy() == 8)[0])
        val_data0 = val_data[ind]
        val_targets0 = val_targets[ind]
        val_set_seg0 = torch.zeros((len(val_targets0), 2, opt.newsize, opt.newsize))
        for i in range(len(val_targets0)):
            u = np.random.uniform(0.0, 1.0)
            if val_targets0[i] == 0:
                if u < 0.8:
                    val_set_seg0[i, 0, :, :] = torch.ones_like(val_data0[i, 0, :, :]) - torch.round(val_data0[i, 0, :, :])
                else:
                    val_set_seg0[i, 1, :, :] = torch.ones_like(val_data0[i, 0, :, :]) - torch.round(val_data0[i, 0, :, :])
            if val_targets0[i] == 8:
                if u < 0.8:
                    val_set_seg0[i, 1, :, :] = torch.ones_like(val_data0[i, 0, :, :]) - torch.round(val_data0[i, 0, :, :])
                else:
                    val_set_seg0[i, 0, :, :] = torch.ones_like(val_data0[i, 0, :, :]) - torch.round(val_data0[i, 0, :, :])
        valset_size = len(val_targets0)  # get the number of images in the dataset.

        print('The number of training images = %d' % dataset_size)
        print('The number of validation images = %d' % valset_size)

        if opt.pretrain == 0:
            model = create_model(opt)
        else:
            if device == 'cuda':
                model = torch.load(opt.pretrained_model_name)
            else:
                model = torch.load(opt.pretrained_model_name, map_location=torch.device('cpu'))
        model = model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=opt.lr)
        total_iters = 0  # the total number of training iterations

        print('Start training')
        for epoch in range(1,
                           opt.n_epochs + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

            train_data = preprocess(train_data0, 1.0, 0.0, opt.x_bins, True)
            train_set_seg = preprocess(train_set_seg0, 1.0, 0.0, opt.y_bins, True)
            val_data = preprocess(val_data0, 1.0, 0.0, opt.x_bins, True)
            val_set_seg = preprocess(val_set_seg0, 1.0, 0.0, opt.y_bins, True)
            dataset = Data.DataLoader(
                Data.TensorDataset(train_data, train_set_seg),
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=int(opt.num_threads))
            val_dataset = Data.DataLoader(
                Data.TensorDataset(val_data, val_set_seg),
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=int(opt.num_threads))

            for i, data in enumerate(dataset):  # inner loop within one epoch
                if device == 'cuda':
                    data = [x.to(device) for x in data]
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                x = data[0].float()
                y = data[1].float()
                z, nll = model.forward(x, y)
                loss = torch.mean(nll)
                model.zero_grad()
                optim.zero_grad()
                loss.backward()
                if opt.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), opt.max_grad_clip)
                if opt.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
                optim.step()

            val_loss = 0
            for i, data in enumerate(val_dataset):
                if device == 'cuda':
                    data = [x.to(device) for x in data]
                x = data[0].float()
                y = data[1].float()
                z, nll = model.forward(x, y)
                valloss = torch.sum(nll)
                val_loss += valloss.detach().cpu().numpy()
            val_loss /= valset_size

            print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss.detach().cpu().numpy()), 'val loss {}'.format(val_loss))

    if opt.sample_method == 1:
        # create a dataset given opt.dataset_mode and other options
        train_data, train_targets, _, _, _, _ = load_data_seg(opt)
        full_size = len(train_targets)
        indices = list(range(full_size))
        split = int(np.floor(opt.val_proportion * full_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        val_data = train_data[val_indices, :, :, :]
        val_targets = train_targets[val_indices]
        #val_set_seg = train_set_seg[val_indices, :, :, :]
        train_data = train_data[train_indices, :, :, :]
        train_targets = train_targets[train_indices]
        #train_set_seg = train_set_seg[train_indices, :, :, :]

        ind = np.union1d(np.where(train_targets.detach().cpu().numpy() == 0)[0],
                         np.where(train_targets.detach().cpu().numpy() == 8)[0])
        train_data0 = train_data[ind]
        train_targets0 = train_targets[ind]
        dataset_size = len(train_targets0)  # get the number of images in the dataset.

        ind = np.union1d(np.where(val_targets.detach().cpu().numpy() == 0)[0],
                         np.where(val_targets.detach().cpu().numpy() == 8)[0])
        val_data0 = val_data[ind]
        val_targets0 = val_targets[ind]
        valset_size = len(val_targets0)  # get the number of images in the dataset.

        print('The number of training images = %d' % dataset_size)
        print('The number of validation images = %d' % valset_size)

        if opt.pretrain == 0:
            model = create_model(opt)
        else:
            if device == 'cuda':
                model = torch.load(opt.pretrained_model_name)
            else:
                model = torch.load(opt.pretrained_model_name, map_location=torch.device('cpu'))
        model = model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=opt.lr)
        total_iters = 0  # the total number of training iterations

        print('Start training')
        for epoch in range(1,
                           opt.n_epochs + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

            train_set_seg0 = torch.zeros((len(train_targets0), 2, opt.newsize, opt.newsize))
            for i in range(len(train_targets0)):
                u = np.random.uniform(0.0, 1.0)
                if train_targets0[i] == 0:
                    if u < 0.8:
                        train_set_seg0[i, 0, :, :] = torch.ones_like(train_data0[i, 0, :, :]) - torch.round(
                            train_data0[i, 0, :, :])
                    else:
                        train_set_seg0[i, 1, :, :] = torch.ones_like(train_data0[i, 0, :, :]) - torch.round(
                            train_data0[i, 0, :, :])
                if train_targets0[i] == 8:
                    if u < 0.8:
                        train_set_seg0[i, 1, :, :] = torch.ones_like(train_data0[i, 0, :, :]) - torch.round(
                            train_data0[i, 0, :, :])
                    else:
                        train_set_seg0[i, 0, :, :] = torch.ones_like(train_data0[i, 0, :, :]) - torch.round(
                            train_data0[i, 0, :, :])

            val_set_seg0 = torch.zeros((len(val_targets0), 2, opt.newsize, opt.newsize))
            for i in range(len(val_targets0)):
                u = np.random.uniform(0.0, 1.0)
                if val_targets0[i] == 0:
                    if u < 0.8:
                        val_set_seg0[i, 0, :, :] = torch.ones_like(val_data0[i, 0, :, :]) - torch.round(
                            val_data0[i, 0, :, :])
                    else:
                        val_set_seg0[i, 1, :, :] = torch.ones_like(val_data0[i, 0, :, :]) - torch.round(
                            val_data0[i, 0, :, :])
                if val_targets0[i] == 8:
                    if u < 0.8:
                        val_set_seg0[i, 1, :, :] = torch.ones_like(val_data0[i, 0, :, :]) - torch.round(
                            val_data0[i, 0, :, :])
                    else:
                        val_set_seg0[i, 0, :, :] = torch.ones_like(val_data0[i, 0, :, :]) - torch.round(
                            val_data0[i, 0, :, :])

            train_data = preprocess(train_data0, 1.0, 0.0, opt.x_bins, True)
            train_set_seg = preprocess(train_set_seg0, 1.0, 0.0, opt.y_bins, True)
            val_data = preprocess(val_data0, 1.0, 0.0, opt.x_bins, True)
            val_set_seg = preprocess(val_set_seg0, 1.0, 0.0, opt.y_bins, True)
            dataset = Data.DataLoader(
                Data.TensorDataset(train_data, train_set_seg),
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=int(opt.num_threads))
            val_dataset = Data.DataLoader(
                Data.TensorDataset(val_data, val_set_seg),
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=int(opt.num_threads))

            for i, data in enumerate(dataset):  # inner loop within one epoch
                if device == 'cuda':
                    data = [x.to(device) for x in data]
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                x = data[0].float()
                y = data[1].float()
                z, nll = model.forward(x, y)
                loss = torch.mean(nll)
                model.zero_grad()
                optim.zero_grad()
                loss.backward()
                if opt.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), opt.max_grad_clip)
                if opt.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
                optim.step()

            val_loss = 0
            for i, data in enumerate(val_dataset):
                if device == 'cuda':
                    data = [x.to(device) for x in data]
                x = data[0].float()
                y = data[1].float()
                z, nll = model.forward(x, y)
                valloss = torch.sum(nll)
                val_loss += valloss.detach().cpu().numpy()
            val_loss /= valset_size

            print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss.detach().cpu().numpy()), 'val loss {}'.format(val_loss))

    print('Training finished')
    torch.save(model, opt.save_model_name)
