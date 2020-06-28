import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from colorization import *
from options.test_options import TestOptions
from dataloader import load_data
from ged_mnist import *
from models.unet_model import *


opt = TestOptions().parse()   # get training options
device = 'cuda' if torch.cuda.is_available() else 'cpu'

p = np.zeros((10, 3))
for i in range(10):
    p[i, 0] = 0.1 + 0.7 * (9 - i) / 9
    p[i, 1] = 0.1
    p[i, 2] = 1 - p[i, 0] - p[i, 1]
opt.p = p
_, _, test_data, test_targets, _ = load_data(opt)  # create a dataset given opt.dataset_mode and other options

model = torch.load('/Users/Haoting/Desktop/MSc_code/savedmodels/model_pix2pix01.pt', map_location=torch.device('cpu'))
opt.model_name = 'pix2pix'
opt.gpu_ids = -1
ged = ged_mnist(test_data, test_targets, p, model, opt)
print(ged)

model = Unet(1, 3, 3).to(device)
model.load_state_dict(torch.load('/Users/Haoting/Desktop/MSc_code/savedmodels/unet_model2_mae.pt'))
model = torch.load('/Users/Haoting/Desktop/MSc_code/savedmodels/model_pix2pix01.pt', map_location=torch.device('cpu'))
opt.model_name = 'pix2pix'
opt.gpu_ids = -1
ged = ged_mnist(test_data, test_targets, p, model, opt)
print(ged)