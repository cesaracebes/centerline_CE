import sys
import torch
from monai.networks.nets import UNet, SwinUNETR
from monai.networks.blocks import UnetOutBlock

def get_model(model_name, in_c=3, n_classes=2, pretrained=False, patch_size=None):
    ## UNET ##
    if model_name == 'small_unet_2d':
        model = UNet(spatial_dims=2, in_channels=in_c, out_channels=n_classes, channels=(16, 32, 64), strides=(2, 2,), num_res_units=2, )
    else:
        sys.exit('not a valid model_name, check utils.get_model.py')

    setattr(model, 'n_classes', n_classes)
    setattr(model, 'patch_size', patch_size)

    return model


