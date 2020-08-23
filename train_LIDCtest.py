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


opt = TrainOptions().parse()   # get training options
opt.model_name = 'cglow'
opt.subset = 0.001
opt.batch_size = 4
opt.input_nc = 1
opt.output_nc = 1
opt.seg = 1
opt.newsize = 128
opt.fixed_indices = False

if opt.model_name == 'cglow':
    opt.x_size = (opt.input_nc, opt.newsize, opt.newsize)
    opt.y_size = (opt.output_nc, opt.newsize, opt.newsize)

# create a dataset given opt.dataset_mode and other options
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = LIDC_IDRI(dataset_location='LIDCdata/')
dataset_size = len(dataset)
if opt.fixed_indices == True:
    train_indices = np.load('/content/drive/My Drive/Colab Notebooks/CondNF_ver1/savedmodel/train_indices.npy')
    val_indices = np.load('/content/drive/My Drive/Colab Notebooks/CondNF_ver1/savedmodel/val_indices.npy')
    test_indices = np.load('/content/drive/My Drive/Colab Notebooks/CondNF_ver1/savedmodel/test_indices.npy')
else:
    dataset_size = int(np.floor(opt.subset * dataset_size))
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_val_indices, test_indices = indices[split:], indices[:split]
    train_val_size = len(train_val_indices)
    split = int(np.floor(0.25 * train_val_size))
    train_indices, val_indices = train_val_indices[split:], train_val_indices[:split]
    #np.save('/content/drive/My Drive/Colab Notebooks/CondNF_ver1/savedmodel/train_indices0.npy', train_indices)
    #np.save('/content/drive/My Drive/Colab Notebooks/CondNF_ver1/savedmodel/val_indices0.npy', val_indices)
    #np.save('/content/drive/My Drive/Colab Notebooks/CondNF_ver1/savedmodel/test_indices0.npy', test_indices)
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = Data.DataLoader(dataset, batch_size=opt.batch_size, sampler=train_sampler)
val_loader = Data.DataLoader(dataset, batch_size=1, sampler=val_sampler)
#test_loader = Data.DataLoader(dataset, batch_size=1, sampler=test_sampler)
print("Number of training/val/test patches:", (len(train_indices), len(val_indices), len(test_indices)))

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
best_val_loss = 10000
for epoch in range(1,
                   opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

    for i, (x, y, _) in enumerate(train_loader):  # inner loop within one epoch
        x = x.to(device)
        y = torch.unsqueeze(y, 1)
        y = y.to(device)
        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        if epoch > opt.n_epochs:
            opt.lr *= opt.lr_decay_rate
            optim = torch.optim.Adam(model.parameters(), lr=opt.lr)
        x = x.float()
        y = y.float()
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

    val_loss = 0
    for i, (x, y, _) in enumerate(val_loader):
        x = x.to(device).float()
        y = torch.unsqueeze(y, 1)
        y = y.to(device).float()
        y = preprocess(y, 1.0, 0.0, opt.y_bins, True)
        z, nll = model.forward(x, y)
        valloss = torch.sum(nll)
        val_loss += valloss.detach().cpu().numpy()
    val_loss /= len(val_indices)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        bestmodel = copy.deepcopy(model)

    print('Epoch {} done, '.format(epoch), 'training loss {}'.format(loss.detach().cpu().numpy()), 'val loss {}'.format(val_loss))

print('Training finished')
torch.save(bestmodel, opt.save_model_name)

