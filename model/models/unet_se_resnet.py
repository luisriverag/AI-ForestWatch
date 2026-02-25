"""
UNetSE_resnet — ResNet50-backbone (MoCo-pretrained) UNet with SE blocks.
Includes ResNetSE_wrapper for adding SE after ResNet stages.
"""

import torch
import torch.nn as nn
from torchvision import models

from .se_blocks import (
    BaseModel, _make_se,
    UNetSE_up_block,
)


class ResNetSE_wrapper(BaseModel):
    """
    Wrapper for ResNet layers that adds SE blocks after each layer
    """
    
    def __init__(self, resnet_layer, output_channels, se_reduction=16, use_se=True):
        super(ResNetSE_wrapper, self).__init__()
        self.resnet_layer = resnet_layer
        self.output_channels = output_channels
        
        # Add SE block after the ResNet layer
        self.use_se = use_se
        self.se = _make_se(output_channels, se_reduction) if use_se else None
    
    def forward(self, x):
        # Pass through ResNet layer
        x = self.resnet_layer(x)
        # Apply SE block if enabled
        if self.use_se:
            x = self.se(x)
        return x


class UNetSE_resnet(BaseModel):
    def __init__(self, num_classes, input_channels=3, se_reduction=16, se_flags=None):
        super().__init__()

        if se_flags is None:
            se_flags = {"input": True, "encoder": True, "decoder": True, "bottleneck": False}
        self.se_flags = se_flags

        # # ImageNet pretrained ResNet50 backbone
        # resnet = models.resnet50(pretrained=True)


        # # # # Input random, rest Moco pretrained ResNet50 backbone
        # # Pretrained ResNet50 backbone
        # resnet = models.resnet50(weights=None)

        # checkpoint = torch.load("Data/PretrainedModel/B13_rn50_moco_0099_ckpt.pth", map_location="cpu")
        # state_dict = checkpoint["state_dict"]

        # # strip possible prefixes like "module.encoder_q."
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.startswith("module.encoder_q."):
        #         k = k.replace("module.encoder_q.", "")
        #     if k.startswith("encoder."):
        #         k = k.replace("encoder.", "")
        #     new_state_dict[k] = v

        # # Create the new first layer for 18 channels (don't load pretrained weights for this)
        # self.layer0_conv = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # # Initialize with Xavier/He initialization
        # nn.init.xavier_uniform_(self.layer0_conv.weight)

        # # Load pretrained weights for the rest of ResNet (skip conv1)
        # pretrained_dict = {k: v for k, v in new_state_dict.items() if k != 'conv1.weight' and k != 'conv1.bias'}
        # resnet.load_state_dict(pretrained_dict, strict=False)

        # # Use the pretrained batch norm and activation from the checkpoint
        # self.layer0_bn = resnet.bn1
        # self.layer0_relu = resnet.relu
        # # SE block after layer0
        # self.layer0_se = _make_se(64, se_reduction) if self.se_flags["encoder"] else None



        # # # Input 13 pretrained 5 random, rest Moco pretrained ResNet50 backbone
        # Pretrained ResNet50 backbone
        resnet = models.resnet50(weights=None)

        checkpoint = torch.load("Data/PretrainedModel/B13_rn50_moco_0099_ckpt.pth", map_location="cpu")
        state_dict = checkpoint["state_dict"]

        # strip possible prefixes like "module.encoder_q."
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module.encoder_q."):
                k = k.replace("module.encoder_q.", "")
            if k.startswith("encoder."):
                k = k.replace("encoder.", "")
            new_state_dict[k] = v

        # Create the new first layer for 18 channels
        self.layer0_conv = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Load pretrained weights for the rest of ResNet (skip conv1)
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k != 'conv1.weight' and k != 'conv1.bias'}
        resnet.load_state_dict(pretrained_dict, strict=False)

        # Now transfer the first 13 channels from pretrained conv1 to our new layer0_conv
        with torch.no_grad():
            # Get the pretrained conv1 weights [64, 13, 7, 7]
            pretrained_conv1 = new_state_dict['conv1.weight']  # [64, 13, 7, 7]
            
            # Copy the first 13 channels to our new layer
            self.layer0_conv.weight.data[:, :13, :, :] = pretrained_conv1
            
            # The remaining 5 channels (13-17) are already randomly initialized by PyTorch
            # You can optionally re-initialize them with a specific method:
            # nn.init.xavier_uniform_(self.layer0_conv.weight[:, 13:, :, :])

        # Use the pretrained batch norm and activation from the checkpoint
        self.layer0_bn = resnet.bn1
        self.layer0_relu = resnet.relu
        # SE block after layer0
        self.layer0_se = _make_se(64, se_reduction) if self.se_flags["encoder"] else None

        self.maxpool = resnet.maxpool
        # self.layer1 = resnet.layer1  # 256 channels
        # self.layer2 = resnet.layer2  # 512 channels
        # self.layer3 = resnet.layer3  # 1024 channels
        # self.layer4 = resnet.layer4  # 2048 channels

        self.layer1 = ResNetSE_wrapper(resnet.layer1, 256, se_reduction, self.se_flags["encoder"])
        self.layer2 = ResNetSE_wrapper(resnet.layer2, 512, se_reduction, self.se_flags["encoder"]) 
        self.layer3 = ResNetSE_wrapper(resnet.layer3, 1024, se_reduction, self.se_flags["encoder"])
        self.layer4 = ResNetSE_wrapper(resnet.layer4, 2048, se_reduction, self.se_flags["encoder"])

        # Bottleneck (x_mid) at the bottom like in UNetSE
        self.activate = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        self.mid_conv_2048_2048_a = nn.Conv2d(2048, 2048, 3, padding=1)
        self.mid_conv_2048_2048_b = nn.Conv2d(2048, 2048, 3, padding=1)

        # # Freeze all encoder layers except layer0
        # for param in self.layer1.parameters():
        #     param.requires_grad = False
        # for param in self.layer2.parameters():
        #     param.requires_grad = False
        # for param in self.layer3.parameters():
        #     param.requires_grad = False
        # for param in self.layer4.parameters():
        #     param.requires_grad = False
        # # self.layer0 is left trainable (default)

        # Freeze only the ResNet layers, not the SE blocks
        for param in self.layer1.resnet_layer.parameters():
            param.requires_grad = False
        for param in self.layer2.resnet_layer.parameters():
            param.requires_grad = False
        for param in self.layer3.resnet_layer.parameters():
            param.requires_grad = False
        for param in self.layer4.resnet_layer.parameters():
            param.requires_grad = False
            
        # Optional SE on input
        self.se_input = _make_se(input_channels, max(2, input_channels // 2)) if self.se_flags["input"] else None

        # Decoder path (use UNet-style skips with a proper bottleneck)
        # decoder4: skip from x3 (1024ch), input from x_mid (2048ch)
        self.decoder4 = UNetSE_up_block(prev_channel=1024, input_channel=2048,
                                        output_channel=512, se_reduction=se_reduction,
                                        use_se=self.se_flags["decoder"])
        # decoder3: skip from x2 (512ch), input from decoder4 output (512ch)
        self.decoder3 = UNetSE_up_block(prev_channel=512, input_channel=512,
                                        output_channel=256, se_reduction=se_reduction,
                                        use_se=self.se_flags["decoder"])
        # decoder2: skip from x1 (256ch), input from decoder3 output (256ch)
        self.decoder2 = UNetSE_up_block(prev_channel=256, input_channel=256,
                                        output_channel=128, se_reduction=se_reduction,
                                        use_se=self.se_flags["decoder"])
        # decoder1: skip from x0 (64ch), input from decoder2 output (128ch)
        self.decoder1 = UNetSE_up_block(prev_channel=64, input_channel=128,
                                        output_channel=64, se_reduction=se_reduction,
                                        use_se=self.se_flags["decoder"])

        # Final upsample to original resolution (no skip at input resolution)
        self.decoder0 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def _maybe_se_input(self, x_in):
        return self.se_input(x_in) if self.se_input else x_in

    def forward(self, x):
        x_in = self._maybe_se_input(x)

        # Encoder (ResNet stages)
        # x0 = self.layer0(x_in)             # 64
        x0 = self.layer0_relu(self.layer0_bn(self.layer0_conv(x_in)))

        if self.layer0_se is not None:
            x0 = self.layer0_se(x0)

        x1 = self.layer1(self.maxpool(x0)) # 256
        x2 = self.layer2(x1)               # 512
        x3 = self.layer3(x2)               # 1024
        x4 = self.layer4(x3)               # 2048

        # Bottleneck
        x_mid = self.mid_conv_2048_2048_a(x4); x_mid = self.activate(x_mid)
        x_mid = self.mid_conv_2048_2048_b(x_mid); x_mid = self.activate(x_mid)
        x_mid = self.dropout(x_mid)

        # Decoder path with skip connections (UNet-style)
        x = self.decoder4(x3, x_mid)
        x = self.decoder3(x2, x)
        x = self.decoder2(x1, x)
        x = self.decoder1(x0, x)
        x = self.decoder0(x)

        x = self.final_conv(x)
        return x, self.softmax(x)
