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
    opt.x_bins = 1.0 / 255.0
    # this p is just to avoid bug for line 38, actually not related to the definition of this example
    p = np.zeros((10, 3))
    for i in range(10):
        p[i, 0] = 0.1 + 0.7 * (9 - i) / 9
        p[i, 1] = 0.1
        p[i, 2] = 1 - p[i, 0] - p[i, 1]
    opt.p = p

    opt.x_size = (opt.input_nc, opt.newsize, opt.newsize)
    opt.y_size = (opt.output_nc, opt.newsize, opt.newsize)

    # load full MNIST data
    x_data, y_data, _, _, _ = load_data(opt)
    # fix one sample 0 and one sample 8
    #ind0 = np.where(y_data.detach().cpu().numpy() == 0)[0]
    #np.random.shuffle(ind0)
    #ind0 = ind0[0]
    #ind8 = np.where(y_data.detach().cpu().numpy() == 8)[0]
    #np.random.shuffle(ind8)
    #ind8 = ind8[0]
    ind0 = 11014
    ind8 = 54341
    dataset_size = 128  # get the number of images in the dataset.
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

    x_0 = x_data[np.array([ind0]), :, :, :]
    y_0_red = colorize_red(x_0).float().to(device)
    y_0_blue = colorize_blue(x_0).float().to(device)
    x_0 = x_0.to(device)
    loss_0_red = np.zeros(int(opt.n_epochs/10))
    loss_0_blue = np.zeros(int(opt.n_epochs/10))
    x_8 = x_data[np.array([ind8]), :, :, :]
    y_8_red = colorize_red(x_8).float().to(device)
    y_8_blue = colorize_blue(x_8).float().to(device)
    x_8 = x_8.to(device)
    loss_8_red = np.zeros(int(opt.n_epochs/10))
    loss_8_blue = np.zeros(int(opt.n_epochs/10))

    print('Start training')
    for epoch in range(1,
                       opt.n_epochs + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        # colorise 64 digit zeros (w.p. 0.8 for red and 0.2 for blue)
        train_data = torch.zeros((dataset_size, 1, 32, 32))
        train_target = torch.zeros((dataset_size, 3, 32, 32))
        for i in range(int(dataset_size / 2)):
            train_data[i, 0, :, :] = x_data[ind0, 0, :, :]
            u = np.random.uniform(0.0, 1.0)
            if u < 0.8:
                train_target[np.array([i]), :, :, :] = colorize_red(train_data[np.array([i]), :, :, :]).float()
            else:
                train_target[np.array([i]), :, :, :] = colorize_blue(train_data[np.array([i]), :, :, :]).float()
        # colorise 64 digit eights (w.p. 0.2 for red and 0.8 for blue)
        for i in range(int(dataset_size / 2), dataset_size):
            train_data[i, 0, :, :] = x_data[ind8, 0, :, :]
            u = np.random.uniform(0.0, 1.0)
            if u > 0.8:
                train_target[np.array([i]), :, :, :] = colorize_red(train_data[np.array([i]), :, :, :]).float()
            else:
                train_target[np.array([i]), :, :, :] = colorize_blue(train_data[np.array([i]), :, :, :]).float()
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        train_data = train_data[indices, :, :, :]
        train_target = train_target[indices, :, :, :]

        dataset = Data.DataLoader(
            Data.TensorDataset(train_data, train_target),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.num_threads))

        # train
        for i, data in enumerate(dataset):  # inner loop within one epoch
            if device == 'cuda':
                data = [x.to(device) for x in data]
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            x = data[0].float()
            y = data[1].float()
            y = preprocess(y, 1.0, 0.0, opt.x_bins, True)
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

        # track the loss for the ground truth target of (0, 8) and (red, blue) respectively
        if epoch % 10 == 0:
            _, nll_1 = model.forward(x_0, preprocess(y_0_red, 1.0, 0.0, opt.x_bins, True))
            _, nll_2 = model.forward(x_0, preprocess(y_0_blue, 1.0, 0.0, opt.x_bins, True))
            _, nll_3 = model.forward(x_8, preprocess(y_8_red, 1.0, 0.0, opt.x_bins, True))
            _, nll_4 = model.forward(x_8, preprocess(y_8_blue, 1.0, 0.0, opt.x_bins, True))
            loss_0_red[int(epoch/10)-1] = nll_1.detach().cpu().numpy()
            loss_0_blue[int(epoch/10)-1] = nll_2.detach().cpu().numpy()
            loss_8_red[int(epoch/10)-1] = nll_3.detach().cpu().numpy()
            loss_8_blue[int(epoch/10)-1] = nll_4.detach().cpu().numpy()

        print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss.detach().cpu().numpy()))
        if epoch % 10 == 0:
            print('loss for red 0 {}'.format(nll_1.detach().cpu().numpy()), 'loss for blue 0 {}'.format(nll_2.detach().cpu().numpy()))
            print('loss for red 8 {}'.format(nll_3.detach().cpu().numpy()), 'loss for blue 8 {}'.format(nll_4.detach().cpu().numpy()))

    print('Training finished')
    torch.save(model, opt.save_model_name)
    np.save('/content/drive/My Drive/Colab Notebooks/debugmodel/model_debug_smallMNIST_001_loss_0_red.npy', loss_0_red)
    np.save('/content/drive/My Drive/Colab Notebooks/debugmodel/model_debug_smallMNIST_001_loss_0_blue.npy', loss_0_blue)
    np.save('/content/drive/My Drive/Colab Notebooks/debugmodel/model_debug_smallMNIST_001_loss_8_red.npy', loss_8_red)
    np.save('/content/drive/My Drive/Colab Notebooks/debugmodel/model_debug_smallMNIST_001_loss_8_blue.npy', loss_8_blue)
