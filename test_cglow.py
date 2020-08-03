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
from dataloader import load_data
from data import *
from models import create_model


opt = TrainOptions().parse()   # get training options
opt.model_name = 'cglow'
opt.subset = 0.0001
opt.batch_size = 2
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

opt.sample_method = 0
opt.seg = 1
opt.output_nc = 1
opt.pretrain = 0
opt.pretrained_model_name = '/Users/Haoting/Desktop/MSc_code/savedmodels/model_cglow002.pt'
opt.n_epochs = 5
opt.n_epochs_decay = 5

if opt.sample_method == 0:
    # create a dataset given opt.dataset_mode and other options
    if opt.seg == 1:
        train_data, train_targets, _, _, _, train_set_seg = load_data_seg(opt)
        dataset_size = len(train_targets)  # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)
        if opt.model_name == 'cglow':
            train_data = preprocess(train_data, 1.0, 0.0, opt.x_bins, True)
            train_set_seg = preprocess(train_set_seg, 1.0, 0.0, opt.x_bins, True)
        dataset = Data.DataLoader(
            Data.TensorDataset(train_data, train_set_seg),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.num_threads))
    else:
        train_data, train_targets, _, _, train_set_colorized = load_data(opt)
        # if opt.model_name == 'cglow':
        #    train_data = torch.repeat_interleave(train_data, 3, dim=1)
        dataset_size = len(train_targets)  # get the number of images in the dataset.
        if opt.model_name == 'cglow':
            train_data = preprocess(train_data, 1.0, 0.0, opt.x_bins, True)
            train_set_colorized = preprocess(train_set_colorized, 1.0, 0.0, opt.x_bins, True)
            # train_set_colorized = preprocess(train_set_colorized, opt.label_scale, opt.label_bias, opt.y_bins, True)
        print('The number of training images = %d' % dataset_size)
        dataset = Data.DataLoader(
            Data.TensorDataset(train_data, train_set_colorized),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.num_threads))

    if opt.pretrain == 0:
        model = create_model(opt)
    else:
        if device == 'cuda':
            model = torch.load(opt.pretrained_model_name)
        else:
            model = torch.load(opt.pretrained_model_name, map_location=torch.device('cpu'))
    if opt.model_name == 'unet':
        model = model.to(device)  # create a model given opt.model and other options
    elif opt.model_name == 'pix2pix' or opt.model_name == 'MSGAN':
        model.netD = model.netD.to(device)
        model.netG = model.netG.to(device)
        model.criterionGAN = model.criterionGAN.to(device)
    elif opt.model_name == 'cglow':
        model = model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=opt.lr)
    else:
        print('Wrong model name')
    total_iters = 0  # the total number of training iterations

    print('Start training')
    for epoch in range(1,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            if device == 'cuda':
                data = [x.to(device) for x in data]
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            if opt.model_name != 'cglow':
                model.set_input(data)
            if opt.model_name == 'unet':
                model.forward()  # calculate loss functions
                loss_print = model.compute_loss()
                model.update_parameters()  # get gradients, update network weights
            if opt.model_name == 'pix2pix':
                lossD_print, lossG_print = model.optimize_parameters()
            if opt.model_name == 'MSGAN':
                lossD_print, lossG_print, losslz_print = model.optimize_parameters()
            if opt.model_name == 'cglow':
                if epoch > opt.n_epochs:
                    opt.lr *= opt.lr_decay_rate
                    optim = torch.optim.Adam(model.parameters(), lr=opt.lr)
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

        if opt.model_name == 'unet':
            if device == 'cuda':
                print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss_print.detach().cpu().numpy()))
            else:
                print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss_print.detach().numpy()))
        if opt.model_name == 'pix2pix':
            if device == 'cuda':
                print('Epoch {} done, '.format(epoch),
                      'discriminator loss {}'.format(lossD_print.detach().cpu().numpy()),
                      'generator loss {}'.format(lossG_print.detach().cpu().numpy()))
            else:
                print('Epoch {} done, '.format(epoch), 'discriminator loss {}'.format(lossD_print.detach().numpy()),
                      'generator loss {}'.format(lossG_print.detach().numpy()))
        if opt.model_name == 'MSGAN':
            if device == 'cuda':
                print('Epoch {} done, '.format(epoch),
                      'discriminator loss {}'.format(lossD_print.detach().cpu().numpy()),
                      'generator loss {}'.format(lossG_print.detach().cpu().numpy()),
                      'mode seeking loss {}'.format(losslz_print.detach().cpu().numpy()))
            else:
                print('Epoch {} done, '.format(epoch), 'discriminator loss {}'.format(lossD_print.detach().numpy()),
                      'generator loss {}'.format(lossG_print.detach().numpy()),
                      'mode seeking loss {}'.format(losslz_print.detach().numpy()))
        if opt.model_name == 'cglow':
            if device == 'cuda':
                print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss.detach().cpu().numpy()))
            else:
                print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss.detach().numpy()))

if opt.sample_method == 1:
    model = create_model(opt)
    if opt.model_name == 'unet':
        model = model.to(device)  # create a model given opt.model and other options
    elif opt.model_name == 'pix2pix' or opt.model_name == 'MSGAN':
        model.netD = model.netD.to(device)
        model.netG = model.netG.to(device)
        model.criterionGAN = model.criterionGAN.to(device)
    elif opt.model_name == 'cglow':
        model = model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=opt.lr)
    else:
        print('Wrong model name')
    total_iters = 0  # the total number of training iterations

    print('Start training')
    for epoch in range(1,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        # create a dataset given opt.dataset_mode and other options
        train_data, train_targets, _, _, train_set_colorized = load_data(opt)
        # if opt.model_name == 'cglow':
        #    train_data = torch.repeat_interleave(train_data, 3, dim=1)
        dataset_size = len(train_targets)  # get the number of images in the dataset.
        if epoch == 1:
            print('The number of training images = %d' % dataset_size)

        if opt.model_name == 'cglow':
            train_data = preprocess(train_data, 1.0, 0.0, opt.x_bins, True)
            train_set_colorized = preprocess(train_set_colorized, 1.0, 0.0, opt.x_bins, True)
            # train_set_colorized = preprocess(train_set_colorized, opt.label_scale, opt.label_bias, opt.y_bins, True)

        dataset = Data.DataLoader(
            Data.TensorDataset(train_data, train_set_colorized),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.num_threads))

        for i, data in enumerate(dataset):  # inner loop within one epoch
            if device == 'cuda':
                data = [x.to(device) for x in data]
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            if opt.model_name != 'cglow':
                model.set_input(data)
            if opt.model_name == 'unet':
                model.forward()  # calculate loss functions
                loss_print = model.compute_loss()
                model.update_parameters()  # get gradients, update network weights
            if opt.model_name == 'pix2pix':
                lossD_print, lossG_print = model.optimize_parameters()
            if opt.model_name == 'MSGAN':
                lossD_print, lossG_print, losslz_print = model.optimize_parameters()
            if opt.model_name == 'cglow':
                if epoch > opt.n_epochs:
                    opt.lr *= opt.lr_decay_rate
                    optim = torch.optim.Adam(model.parameters(), lr=opt.lr)
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

        if opt.model_name == 'unet':
            if device == 'cuda':
                print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss_print.detach().cpu().numpy()))
            else:
                print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss_print.detach().numpy()))
        if opt.model_name == 'pix2pix':
            if device == 'cuda':
                print('Epoch {} done, '.format(epoch),
                      'discriminator loss {}'.format(lossD_print.detach().cpu().numpy()),
                      'generator loss {}'.format(lossG_print.detach().cpu().numpy()))
            else:
                print('Epoch {} done, '.format(epoch), 'discriminator loss {}'.format(lossD_print.detach().numpy()),
                      'generator loss {}'.format(lossG_print.detach().numpy()))
        if opt.model_name == 'MSGAN':
            if device == 'cuda':
                print('Epoch {} done, '.format(epoch),
                      'discriminator loss {}'.format(lossD_print.detach().cpu().numpy()),
                      'generator loss {}'.format(lossG_print.detach().cpu().numpy()),
                      'mode seeking loss {}'.format(losslz_print.detach().cpu().numpy()))
            else:
                print('Epoch {} done, '.format(epoch), 'discriminator loss {}'.format(lossD_print.detach().numpy()),
                      'generator loss {}'.format(lossG_print.detach().numpy()),
                      'mode seeking loss {}'.format(losslz_print.detach().numpy()))
        if opt.model_name == 'cglow':
            if device == 'cuda':
                print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss.detach().cpu().numpy()))
            else:
                print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss.detach().numpy()))

print('Training finished')
