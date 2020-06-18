from models.unet_model import Unet
from models.pix2pix_model import Pix2PixModel
from models.msgan_model import MSGAN

def create_model(opt):
    """Create a model given the option.
    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'
    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    if opt.model_name == 'unet':
        return Unet(opt)
    elif opt.model_name == 'pix2pix':
        return Pix2PixModel(opt)
    elif opt.model_name == 'MSGAN':
        return MSGAN(opt)
    else:
        print("No model with the defined name")
        raise
