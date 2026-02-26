"""
Shared SE building blocks used by all UNet-SE model variants.
"""

import torch
import torch.nn as nn
from torchvision.ops import SqueezeExcitation

try:
    from base.base_model import BaseModel
except ImportError:
    try:
        from base import BaseModel
    except ImportError:
        # Fallback: define BaseModel as nn.Module if not found
        class BaseModel(nn.Module):
            def __init__(self):
                super(BaseModel, self).__init__()


def _make_se(c, reduction=16):
    # safe "squeeze_channels" per torchvision docs: usually c // reduction
    return SqueezeExcitation(input_channels=c, squeeze_channels=max(1, c // reduction))


class MFFBlock(nn.Module):
    """
    Multi-Feature Fusion (MFF) block.
    Takes features from all encoder stages, resizes them to the target resolution,
    concatenates, and fuses with convs.
    """
    def __init__(self, in_channels_list, out_channels):
        super(MFFBlock, self).__init__()
        self.out_channels = out_channels

        # project all inputs to out_channels with 1x1 conv before resizing
        self.proj_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])

        # fusion convs after concatenation
        self.conv1x1 = nn.Conv2d(len(in_channels_list) * out_channels,
                                 out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels,
                                 kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature_maps, target_size):
        """
        feature_maps: list of encoder feature maps [x1, x2, x3, x4]
        target_size: (H, W) resolution to which all maps are resized
        """
        outs = []
        for proj, fmap in zip(self.proj_convs, feature_maps):
            x = proj(fmap)
            x = nn.functional.interpolate(x, size=target_size,
                                          mode="bilinear", align_corners=False)
            outs.append(x)
        fused = torch.cat(outs, dim=1)
        fused = self.relu(self.bn1(self.conv1x1(fused)))
        fused = self.relu(self.bn2(self.conv3x3(fused)))
        return fused 


class UNetSE_down_block(BaseModel):
    """
        Encoder block + optional SE
    """

    def __init__(self, input_channel, output_channel, conv_1=None, conv_2=None, se_reduction=16, use_se=True):
        super(UNetSE_down_block, self).__init__()
        if conv_1:
            print('LOG: Using pretrained convolutional layer', conv_1)
        if conv_2:
            print('LOG: Using pretrained convolutional layer', conv_2)
        self.input_channels = input_channel
        self.output_channels = output_channel

        self.conv1 = conv_1 if conv_1 else nn.Conv2d(
            input_channel, output_channel, kernel_size=3, padding=1)
        self.conv2 = conv_2 if conv_2 else nn.Conv2d(
            output_channel, output_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channel)
        self.bn2 = nn.BatchNorm2d(num_features=output_channel)
        self.activate = nn.ReLU(inplace=True)

        # NEW: SE after the block has produced output_channel features
        self.use_se = use_se
        self.se = _make_se(output_channel, se_reduction) if use_se else None

    def forward(self, x):
        x = self.activate(self.bn1(self.conv1(x)))
        x = self.activate(self.bn2(self.conv2(x)))
        # NEW: SE after the block has produced output_channel features
        if self.use_se:
            x = self.se(x)
        return x


class UNetSE_up_block(BaseModel):
    """
        Decoder block + optional SE
    """

    def __init__(self, prev_channel, input_channel, output_channel, se_reduction=16, use_se=True):
        super(UNetSE_up_block, self).__init__()
        self.output_channels = output_channel
        self.tr_conv_1 = nn.ConvTranspose2d(
            input_channel, input_channel, kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(
            prev_channel + input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(
            output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channel)
        self.bn2 = nn.BatchNorm2d(num_features=output_channel)
        self.activate = nn.ReLU(inplace=True)

        # NEW: SE after the two convs in the decoder block
        self.use_se = use_se
        self.se = _make_se(output_channel, se_reduction) if use_se else None

    def forward(self, prev_feature_map, x):
        x = self.tr_conv_1(x)
        x = self.activate(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.activate(self.bn1(self.conv_1(x)))
        x = self.activate(self.bn2(self.conv_2(x)))
        # NEW: SE after the two convs in the decoder block
        if self.use_se:
            x = self.se(x)
        return x


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, use_se=True, se_reduction=16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k, stride=1, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.se = _make_se(out_ch, se_reduction) if use_se else None

    def forward(self, x):
        x = self.block(x)
        if self.se is not None:
            x = self.se(x)
        return x
