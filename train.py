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
    train_data, train_targets, _, _, train_set_colorized = load_data(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_targets)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    dataset = Data.DataLoader(
            Data.TensorDataset(train_data, train_set_colorized),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.num_threads))
    model = create_model(opt)
    if opt.model_name == 'unet':
        model = model.to(device)      # create a model given opt.model and other options
    elif opt.model_name == 'pix2pix' or opt.model_name == 'MSGAN':
        model.netD = model.netD.to(device)
        model.netG = model.netG.to(device)
        model.criterionGAN = model.criterionGAN.to(device)
    else:
        print('Wrong model name')
    total_iters = 0                # the total number of training iterations

    print('Start training')
    for epoch in range(1, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            if device == 'cuda':
                data = [x.to(device) for x in data]
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            if opt.model_name == 'unet':
                model.forward()     # calculate loss functions
                loss_print = model.compute_loss()
                model.update_parameters()    # get gradients, update network weights
            if opt.model_name == 'pix2pix':
                lossD_print, lossG_print = model.optimize_parameters()
            if opt.model_name == 'MSGAN':
                lossD_print, lossG_print, losslz_print = model.optimize_parameters()

        if opt.model_name == 'unet':
            if device == 'cuda':
                print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss_print.detach().cpu().numpy()))
            else:
                print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss_print.numpy()))
        if opt.model_name == 'pix2pix':
            if device == 'cuda':
                print('Epoch {} done, '.format(epoch), 'discriminator loss {}'.format(lossD_print.detach().cpu().numpy()), 'generator loss {}'.format(lossG_print.detach().cpu().numpy()))
            else:
                print('Epoch {} done, '.format(epoch), 'discriminator loss {}'.format(lossD_print.numpy()), 'generator loss {}'.format(lossG_print.numpy()))
        if opt.model_name == 'MSGAN':
            if device == 'cuda':
                print('Epoch {} done, '.format(epoch), 'discriminator loss {}'.format(lossD_print.detach().cpu().numpy()), 'generator loss {}'.format(lossG_print.detach().cpu().numpy()), 'mode seeking loss {}'.format(losslz_print.detach().cpu().numpy()))
            else:
                print('Epoch {} done, '.format(epoch), 'discriminator loss {}'.format(lossD_print.numpy()), 'generator loss {}'.format(lossG_print.numpy()), 'mode seeking loss {}'.format(losslz_print.numpy()))

    print('Training finished')
    torch.save(model, opt.save_model_name)
