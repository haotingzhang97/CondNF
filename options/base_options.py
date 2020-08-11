import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time."""

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataset_name', type=str, default='MNIST', help='dataset name')
        parser.add_argument('--subset', type=float, default=0.1, help='the percentage of training data loaded')
        parser.add_argument('--val_proportion', type=float, default=0.2, help='the percentage of training data used for validation')
        parser.add_argument('--resize', type=str, default=True, help='whether resize data')
        parser.add_argument('--newsize', type=int, default=32, help='the new size of each side, if resize==True')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # general model parameters
        parser.add_argument('--model_name', type=str, default='unet', help='chooses which model to use. [unet | pix2pix | MSGAN | CNF]')
        parser.add_argument('--seg', type=int, default=1, help='whether treat as a segmentation problem (1) or not (0)')
        # model parameters for all models
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        # model parameters for unet and pix2pix
        parser.add_argument('--num_downs', type=int, default=3, help='# of downsamplings in Unet')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='unet_128', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        # extra parameters for mode seeking pix2pix
        parser.add_argument('--nz', type=int, default=8, help='#latent vector')
        parser.add_argument('--nl', type=str, default='relu', help='non-linearity activation: relu | lrelu | elu')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight on D loss. D(G(A, E(B)))')
        parser.add_argument('--lambda_ms', type=float, default=1.0, help='weight on mode seeking loss')
        # model parameters for cGlow
        #parser.add_argument("--x_size", type=tuple, default=(1, 32, 32))
        #parser.add_argument("--y_size", type=tuple, default=(1, 32, 32))
        parser.add_argument("--x_hidden_channels", type=int, default=128)
        parser.add_argument("--x_hidden_size", type=int, default=32)
        parser.add_argument("--y_hidden_channels", type=int, default=256)
        parser.add_argument("-K", "--flow_depth", type=int, default=8)
        parser.add_argument("-L", "--num_levels", type=int, default=3)
        parser.add_argument("--learn_top", type=bool, default=False)
        parser.add_argument("--max_grad_clip", type=float, default=5)
        parser.add_argument("--max_grad_norm", type=float, default=0)
        # Dataset preprocess parameters for cGlow
        parser.add_argument("--label_scale", type=float, default=1)
        parser.add_argument("--label_bias", type=float, default=0.5)
        parser.add_argument("--x_bins", type=float, default=1.0/256.0)
        parser.add_argument("--y_bins", type=float, default=0.1)

        # dataset parameters
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        #model_name = opt.model
        #model_option_setter = models.get_option_setter(model_name)
        #parser = model_option_setter(parser, self.isTrain)
        #opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        #dataset_name = opt.dataset_mode
        #dataset_option_setter = data.get_option_setter(dataset_name)
        #parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        #if opt.suffix:
        #    suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        #    opt.name = opt.name + suffix

        #self.print_options(opt)

        # set gpu ids
        #str_ids = opt.gpu_ids.split(',')
        #opt.gpu_ids = []
        #for str_id in str_ids:
        #    id = int(str_id)
        #    if id >= 0:
        #        opt.gpu_ids.append(id)
        #if len(opt.gpu_ids) > 0:
        #    torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt


