import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from colorization import *
from options.test_options import TestOptions
from dataloader import load_data


if __name__ == '__main__':
    opt = TestOptions().parse()   # get training options
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    p = np.zeros((10, 3))
    for i in range(10):
        p[i, 0] = 0.1 + 0.7 * (9 - i) / 9
        p[i, 1] = 0.1
        p[i, 2] = 1 - p[i, 0] - p[i, 1]
    opt.p = p
    _, _, test_data, test_targets, _ = load_data(opt)  # create a dataset given opt.dataset_mode and other options

    model = torch.load(opt.load_model_name)
    print('Visualise on test set')
    if opt.model_name == 'unet':
        for digit in range(10):
            plt.figure()
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ind = np.where(test_targets == digit)[0]
            i = ind[np.random.randint(0, len(ind), 1)]
            input = test_data[i, :, :, :].to(device).float()
            input0 = input.detach().cpu().numpy()
            meanlabel = p[digit, 0] * colorize_red(input0, 1) + p[digit, 1] * colorize_green(input0, 1) + p[
                digit, 2] * colorize_blue(input0, 1)
            if np.argmax(p[digit, :]) == 0:
                modelabel = colorize_red(input0, 1)
            elif np.argmax(p[digit, :]) == 1:
                modelabel = colorize_green(input0, 1)
            else:
                modelabel = colorize_blue(input0, 1)
            fig0 = np.swapaxes(np.swapaxes(np.squeeze(meanlabel), 0, 1), 1, 2)
            ax1.imshow(fig0)
            fig0 = np.swapaxes(np.swapaxes(np.squeeze(modelabel), 0, 1), 1, 2)
            ax2.imshow(fig0)
            model.isTrain = False
            model.set_input(input)
            if device == 'cuda':
                output = model.forward().detach().cpu().numpy()
            else:
                output = model.forward().numpy()
            fig0 = np.swapaxes(np.swapaxes(np.squeeze(output), 0, 1), 1, 2)
            ax3.imshow(fig0)
    if opt.model_name == 'pix2pix' or 'MSGAN':
        for digit in range(10):
            plt.figure()
            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1, 7)
            ind = np.where(test_targets == digit)[0]
            i = ind[np.random.randint(0, len(ind), 1)]
            input = test_data[i, :, :, :].to(device).float()
            input0 = input.detach().cpu().numpy()
            meanlabel = p[digit, 0] * colorize_red(input0, 1) + p[digit, 1] * colorize_green(input0, 1) + p[
                digit, 2] * colorize_blue(input0, 1)
            if np.argmax(p[digit, :]) == 0:
                modelabel = colorize_red(input0, 1)
            elif np.argmax(p[digit, :]) == 1:
                modelabel = colorize_green(input0, 1)
            else:
                modelabel = colorize_blue(input0, 1)
            fig0 = np.swapaxes(np.swapaxes(np.squeeze(meanlabel), 0, 1), 1, 2)
            ax1.imshow(fig0)
            fig0 = np.swapaxes(np.swapaxes(np.squeeze(modelabel), 0, 1), 1, 2)
            ax2.imshow(fig0)
            model.isTrain = False
            model.set_input(input)
            if device == 'cuda':
                output = model.forward().detach().cpu().numpy()
            else:
                output = model.forward().numpy()
            fig0 = np.swapaxes(np.swapaxes(np.squeeze(output), 0, 1), 1, 2)
            ax3.imshow(fig0)
            if device == 'cuda':
                output = model.forward().detach().cpu().numpy()
            else:
                output = model.forward().numpy()
            fig0 = np.swapaxes(np.swapaxes(np.squeeze(output), 0, 1), 1, 2)
            ax4.imshow(fig0)
            if device == 'cuda':
                output = model.forward().detach().cpu().numpy()
            else:
                output = model.forward().numpy()
            fig0 = np.swapaxes(np.swapaxes(np.squeeze(output), 0, 1), 1, 2)
            ax5.imshow(fig0)
            if device == 'cuda':
                output = model.forward().detach().cpu().numpy()
            else:
                output = model.forward().numpy()
            fig0 = np.swapaxes(np.swapaxes(np.squeeze(output), 0, 1), 1, 2)
            ax6.imshow(fig0)
            if device == 'cuda':
                output = model.forward().detach().cpu().numpy()
            else:
                output = model.forward().numpy()
            fig0 = np.swapaxes(np.swapaxes(np.squeeze(output), 0, 1), 1, 2)
            ax7.imshow(fig0)