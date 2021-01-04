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
        p[i, 0] = 0.1 + 0.8 * (9 - i) / 9
        p[i, 1] = 0.0
        p[i, 2] = 1 - p[i, 0] - p[i, 1]
    opt.p = p

    if opt.model_name == 'cglow':
        opt.x_size = (opt.input_nc, opt.newsize, opt.newsize)
        opt.y_size = (opt.output_nc, opt.newsize, opt.newsize)

    # create a dataset given opt.dataset_mode and other options
    train_data, train_targets, _, _, _, train_set_seg = load_data_seg(opt)

    full_size = len(train_targets)
    indices = list(range(full_size))
    split = int(np.floor(opt.val_proportion * full_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    val_data = train_data[val_indices, :, :, :]
    val_targets = train_targets[val_indices]
    val_set_seg = train_set_seg[val_indices, :, :, :]
    train_data = train_data[train_indices, :, :, :]
    train_targets = train_targets[train_indices]
    train_set_seg = train_set_seg[train_indices, :, :, :]

    val_dataset = Data.DataLoader(
        Data.TensorDataset(val_data, val_set_seg),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.num_threads))
    valset_size = len(val_indices)
    dataset_size = len(train_indices)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    print('The number of validation images = %d' % valset_size)

    if opt.model_name == 'cglow':
        train_data = preprocess(train_data, 1.0, 0.0, opt.x_bins, True)
        train_set_seg = preprocess(train_set_seg, 1.0, 0.0, opt.y_bins, True)
    dataset = Data.DataLoader(
        Data.TensorDataset(train_data, train_set_seg),
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
    elif opt.model_name == 'cglow':
        model = model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=opt.lr)
    else:
        print('Wrong model name')
    total_iters = 0  # the total number of training iterations

    print('Start training')
    best_val_loss = 10000
    for epoch in range(1,
                       opt.n_epochs + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
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
            if opt.model_name == 'cglow':
                x = data[0].float()
                y = data[1].float()
                z, nll = model.forward(x, y, sigmoid=opt.sigmoid, linear_map=opt.linear_map)
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
            if opt.model_name == 'cglow':
                x = data[0].float()
                y = data[1].float()
                z, nll = model.forward(x, y, sigmoid=opt.sigmoid, linear_map=opt.linear_map)
                valloss = torch.sum(nll)
                val_loss += valloss.detach().cpu().numpy()
        val_loss /= valset_size
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bestmodel = copy.deepcopy(model)

        if opt.model_name == 'unet':
            print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss_print.detach().cpu().numpy()))
        if opt.model_name == 'cglow':
            print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss.detach().cpu().numpy()),
                  'val loss {}'.format(val_loss))

    print('Training finished')
    torch.save(bestmodel, opt.save_model_name)