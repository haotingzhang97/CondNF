import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from colorization import *
from options.test_options import TestOptions
from data import *
from ged_mnist import *
from models.unet_model import *
from models import create_model


opt = TestOptions().parse()   # get training options
device = 'cuda' if torch.cuda.is_available() else 'cpu'

p = np.zeros((10, 3))
for i in range(10):
    p[i, 0] = 0.1 + 0.7 * (9 - i) / 9
    p[i, 1] = 0.1
    p[i, 2] = 1 - p[i, 0] - p[i, 1]
opt.p = p
opt.newsize = 64
_, _, test_data, test_targets, _ = load_data(opt)  # create a dataset given opt.dataset_mode and other options

'''
opt.model_name = 'unet'
model = create_model(opt)
model.load_state_dict(torch.load('/Users/Haoting/Desktop/MSc_code/savedmodels/unet_model2_mae2.pt',map_location=torch.device('cpu')))
opt.gpu_ids = -1
ged = ged_mnist(test_data, test_targets, p, model, opt, 10, 10, 'mssim')
print(ged)
print(np.mean(ged))

model = torch.load('/Users/Haoting/Desktop/MSc_code/savedmodels/model_pix2pix01.pt', map_location=torch.device('cpu'))
opt.model_name = 'pix2pix'
opt.gpu_ids = -1
ged = ged_mnist(test_data, test_targets, p, model, opt, 10, 10, 'mssim')
print(ged)
print(np.mean(ged))

model = torch.load('/Users/Haoting/Desktop/MSc_code/savedmodels/model_msgan07.pt', map_location=torch.device('cpu'))
opt.model_name = 'MSGAN'
opt.gpu_ids = -1
ged = ged_mnist(test_data, test_targets, p, model, opt, 10, 10, 'L1')
print(ged)
print(np.mean(ged))
'''
model = torch.load('/Users/Haoting/Desktop/MSc_code/savedmodels/model_cglow07.pt', map_location=torch.device('cpu'))
opt.model_name = 'cglow'
opt.gpu_ids = -1
ged = ged_mnist(test_data, test_targets, p, model, opt, 10, 10, 'mssim')
print(ged)
print(np.mean(ged))
