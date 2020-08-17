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


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    if opt.model_name == 'cglow':
        opt.x_size = (opt.input_nc, opt.newsize, opt.newsize)
        opt.y_size = (opt.output_nc, opt.newsize, opt.newsize)

    # create a dataset given opt.dataset_mode and other options
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = LIDC_IDRI(dataset_location='LIDCdata/')
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = Data.DataLoader(dataset, batch_size=opt.batch_size, sampler=train_sampler)
    test_loader = Data.DataLoader(dataset, batch_size=1, sampler=test_sampler)
    print("Number of training/test patches:", (len(train_indices), len(test_indices)))

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
    best_val_loss = 10000
    for epoch in range(1,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(train_loader):  # inner loop within one epoch
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
                y = torch.unsqueeze(y, 1)
                y = preprocess(y, 1.0, 0.0, opt.y_bins, True)
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

        # val_loss = 0
        # for i, data in enumerate(val_dataset):
        #    if device == 'cuda':
        #        data = [x.to(device) for x in data]
        #    if opt.model_name == 'cglow':
        #        x = data[0].float()
        #        y = data[1].float()
        #        z, nll = model.forward(x, y)
        #        valloss = torch.sum(nll)
        #        val_loss += valloss.detach().cpu().numpy()
        # val_loss /= valset_size
        # if val_loss < best_val_loss:
        #    best_val_loss = val_loss
        #    bestmodel = copy.deepcopy(model)

        if opt.model_name == 'unet':
            print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss_print.detach().cpu().numpy()))
        if opt.model_name == 'pix2pix':
            print('Epoch {} done, '.format(epoch),
                  'discriminator loss {}'.format(lossD_print.detach().cpu().numpy()),
                  'generator loss {}'.format(lossG_print.detach().cpu().numpy()))
        if opt.model_name == 'MSGAN':
            print('Epoch {} done, '.format(epoch),
                  'discriminator loss {}'.format(lossD_print.detach().cpu().numpy()),
                  'generator loss {}'.format(lossG_print.detach().cpu().numpy()),
                  'mode seeking loss {}'.format(losslz_print.detach().cpu().numpy()))
        if opt.model_name == 'cglow':
            print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss.detach().cpu().numpy()))  # ,
        #          'val loss {}'.format(val_loss))

    print('Training finished')
    torch.save(model, opt.save_model_name)
    # torch.save(bestmodel, opt.save_model_name)
