# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
    UNet model definition in here
"""

import torch
import torch.nn as nn
from base import BaseModel
from torch.optim import *
from torchvision import models
from .models.unet import UNet as UNetModel
from .models.unet_se import UNetSE as UNetSEModel
from .models.unet_mff import UNetMFF as UNetMFFModel
from .models.segformer import CustomSegformer as CustomSegformerModel

def UNet(input_channels, num_classes, topology="ENC_4_DEC_4"):
    """
    UNet model factory function
    """
    return UNetModel(topology=topology, input_channels=input_channels, num_classes=num_classes)


def UNetSE(input_channels, num_classes, topology="ENC_4_DEC_4", se_reduction=16, se_flags=None):
    """
    UNetSE model factory function
    """
    return UNetSEModel(topology=topology, input_channels=input_channels, num_classes=num_classes, se_reduction=se_reduction, se_flags=se_flags)


def UNetMFF(input_channels, num_classes, topology="ENC_4_DEC_4"):
    """
    UNetMFF model factory function
    """
    return UNetMFFModel(topology=topology, input_channels=input_channels, num_classes=num_classes)


def CustomSegformer(input_channels, num_classes, base_model='nvidia/mit-b0'):
    """
    CustomSegformer model factory function
    """
    return CustomSegformerModel(input_channels=input_channels, num_classes=num_classes, base_model=base_model)


@torch.no_grad()
def check_model(model_type="UNet", topology="ENC_1_DEC_1", input_channels=7, num_classes=2, input_shape=[4, 7, 64, 64], base_model='nvidia/mit-b0', se_reduction=16, se_flags=None):
    
    if model_type == "UNet":
        model = UNet(topology=topology, input_channels=input_channels, num_classes=num_classes)
    elif model_type == "UNetSE":
        model = UNetSE(topology=topology, input_channels=input_channels, num_classes=num_classes, se_reduction=se_reduction, se_flags=se_flags)
    elif model_type == "UNetMFF":
        model = UNetMFF(topology=topology, input_channels=input_channels, num_classes=num_classes)
    elif model_type == "CustomSegformer":
        model = CustomSegformer(input_channels=input_channels, num_classes=num_classes, base_model=base_model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    in_tensor = torch.Tensor(*input_shape)
    with torch.no_grad():
        out_tensor, softmaxed = model(in_tensor)
        print(in_tensor.shape, out_tensor.shape)


if __name__ == '__main__':
    # check_model
    check_model(topology="ENC_1_DEC_1", input_channels=7,
                num_classes=2, input_shape=[4, 7, 64, 64])
