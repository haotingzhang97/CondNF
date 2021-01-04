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
    opt.x_size = (opt.input_nc, opt.newsize, opt.newsize)
    opt.y_size = (opt.output_nc, opt.newsize, opt.newsize)
    if opt.sample_method == 0:
        # create a dataset given opt.dataset_mode and other options
        dataset_size = 1000

        train_data = torch.zeros((dataset_size, 1, 8, 8))
        train_indices = list(range(dataset_size))
        split = int(np.floor(opt.val_proportion * dataset_size))
        np.random.shuffle(train_indices)

        train_label = torch.zeros_like(train_data)
        for i in range(dataset_size):
            u = np.random.uniform(0.0, 1.0)
            if u < 0.8:
                train_label[i, 0, :, :] = torch.normal(mean=torch.ones((8, 8)), std=0.1*torch.ones((8, 8)))
            else:
                train_label[i, 0, :, :] = torch.normal(mean=-torch.ones((8, 8)), std=0.1*torch.ones((8, 8)))

        print('The number of training images = %d' % dataset_size)

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

            dataset = Data.DataLoader(
                Data.TensorDataset(train_data, train_label),
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

            print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss.detach().cpu().numpy()))

    if opt.sample_method == 1:
        # create a dataset given opt.dataset_mode and other options
        train_data = torch.zeros((1000, 1, 8, 8))
        dataset_size = 1000
        train_indices = list(range(dataset_size))
        split = int(np.floor(opt.val_proportion * dataset_size))
        np.random.shuffle(train_indices)

        print('The number of training images = %d' % dataset_size)

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

            train_label = torch.zeros_like(train_data)
            for i in range(dataset_size):
                u = np.random.uniform(0.0, 1.0)
                if u < 0.8:
                    train_label[i, 0, :, :] = torch.normal(mean=torch.ones((8, 8)), std=0.1 * torch.ones((8, 8)))
                else:
                    train_label[i, 0, :, :] = torch.normal(mean=-torch.ones((8, 8)), std=0.1 * torch.ones((8, 8)))

            dataset = Data.DataLoader(
                Data.TensorDataset(train_data, train_label),
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

            print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss.detach().cpu().numpy()))

    print('Training finished')
    torch.save(model, opt.save_model_name)