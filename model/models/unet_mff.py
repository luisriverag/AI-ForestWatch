import torch
import torch.nn as nn
from base import BaseModel
from torch.optim import *
from torchvision import models
import timm

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


# Import from unet.py instead of redefining
from .unet import UNet_down_block, UNet_up_block


class UNetMFF(BaseModel):
    def __init__(self, topology, input_channels, num_classes):
        super(UNetMFF, self).__init__()
        # these topologies are possible right now
        self.topologies = {
            "ENC_1_DEC_1": self.ENC_1_DEC_1,
            "ENC_2_DEC_2": self.ENC_2_DEC_2,
            "ENC_3_DEC_3": self.ENC_3_DEC_3,
            "ENC_4_DEC_4": self.ENC_4_DEC_4,
        }
        assert topology in self.topologies
        vgg_trained = models.vgg11(pretrained=True)
        pretrained_layers = list(vgg_trained.features)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.6)
        self.activate = nn.ReLU()
        self.encoder_1 = UNet_down_block(input_channels, 64)
        self.encoder_2 = UNet_down_block(64, 128, conv_1=pretrained_layers[3])
        self.encoder_3 = UNet_down_block(
            128, 256, conv_1=pretrained_layers[6], conv_2=pretrained_layers[8])
        self.encoder_4 = UNet_down_block(
            256, 512, conv_1=pretrained_layers[11], conv_2=pretrained_layers[13])
        self.mid_conv_64_64_a = nn.Conv2d(64, 64, 3, padding=1)
        self.mid_conv_64_64_b = nn.Conv2d(64, 64, 3, padding=1)
        self.mid_conv_128_128_a = nn.Conv2d(128, 128, 3, padding=1)
        self.mid_conv_128_128_b = nn.Conv2d(128, 128, 3, padding=1)
        self.mid_conv_256_256_a = nn.Conv2d(256, 256, 3, padding=1)
        self.mid_conv_256_256_b = nn.Conv2d(256, 256, 3, padding=1)
        self.mid_conv_512_1024 = nn.Conv2d(512, 1024, 3, padding=1)
        self.mid_conv_1024_1024 = nn.Conv2d(1024, 1024, 3, padding=1)
        # MFF blocks: fuse all encoder features at each decoder level's resolution
        # Each MFFBlock receives features from all 4 encoder stages [64, 128, 256, 512]
        # and outputs a fixed-width (64-channel) fused skip connection
        enc_channels = [64, 128, 256, 512]
        mff_out = 64  # match decoder_1 output / encoder_1 output channels
        self.mff1 = MFFBlock(enc_channels, mff_out)
        self.mff2 = MFFBlock(enc_channels, mff_out)
        self.mff3 = MFFBlock(enc_channels, mff_out)
        self.mff4 = MFFBlock(enc_channels, mff_out)

        # prev_channel=mff_out (64) because MFF blocks output 64-channel fused skips
        self.decoder_4 = UNet_up_block(prev_channel=mff_out,
                                       input_channel=self.mid_conv_1024_1024.out_channels, output_channel=256)
        self.decoder_3 = UNet_up_block(prev_channel=mff_out,
                                       input_channel=self.decoder_4.output_channels, output_channel=128)
        self.decoder_2 = UNet_up_block(prev_channel=mff_out,
                                       input_channel=self.decoder_3.output_channels, output_channel=64)
        self.decoder_1 = UNet_up_block(prev_channel=mff_out,
                                       input_channel=self.decoder_2.output_channels, output_channel=64)

        self.binary_last_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.forward = self.topologies[topology]
        print('\n\n' + "#" * 100)
        print("(LOG): The following Model Topology will be Utilized: {}".format(
            self.forward.__name__))
        print("#" * 100 + '\n\n')

    def ENC_1_DEC_1(self, x_in):
        x1_cat = self.encoder_1(x_in)
        x1_cat_1 = self.dropout(x1_cat)
        x1 = self.max_pool(x1_cat_1)

        x_mid = self.activate(self.mid_conv_64_64_a(x1))
        x_mid = self.activate(self.mid_conv_64_64_b(x_mid))
        x_mid = self.dropout(x_mid)

        # MFF skip for level 1
        mff1_out = self.mff1([x1_cat, x1_cat, x1_cat, x1_cat], target_size=x1_cat.shape[2:])

        x = self.decoder_1(mff1_out, x_mid)
        x = self.binary_last_conv(x)
        return x, self.softmax(x)

    def ENC_2_DEC_2(self, x_in):
        x1_cat = self.encoder_1(x_in)
        x1 = self.max_pool(x1_cat)

        x2_cat = self.encoder_2(x1)
        x2_cat_1 = self.dropout(x2_cat)
        x2 = self.max_pool(x2_cat_1)

        x_mid = self.activate(self.mid_conv_128_128_a(x2))
        x_mid = self.activate(self.mid_conv_128_128_b(x_mid))
        x_mid = self.dropout(x_mid)

        # MFF skips (pad to 4 inputs to match MFFBlock's enc_channels [64,128,256,512])
        mff1_out = self.mff1([x1_cat, x2_cat, x2_cat, x2_cat], target_size=x1_cat.shape[2:])
        mff2_out = self.mff2([x1_cat, x2_cat, x2_cat, x2_cat], target_size=x2_cat.shape[2:])

        x = self.decoder_2(mff2_out, x_mid)
        x = self.decoder_1(mff1_out, x)
        x = self.binary_last_conv(x)
        return x, self.softmax(x)

    def ENC_3_DEC_3(self, x_in):
        x1_cat = self.encoder_1(x_in)
        x1 = self.max_pool(x1_cat)

        x2_cat = self.encoder_2(x1)
        x2_cat_1 = self.dropout(x2_cat)
        x2 = self.max_pool(x2_cat_1)

        x3_cat = self.encoder_3(x2)
        x3 = self.max_pool(x3_cat)

        x_mid = self.activate(self.mid_conv_256_256_a(x3))
        x_mid = self.activate(self.mid_conv_256_256_b(x_mid))
        x_mid = self.dropout(x_mid)

        # MFF skips (pad to 4 inputs to match MFFBlock's enc_channels [64,128,256,512])
        mff1_out = self.mff1([x1_cat, x2_cat, x3_cat, x3_cat], target_size=x1_cat.shape[2:])
        mff2_out = self.mff2([x1_cat, x2_cat, x3_cat, x3_cat], target_size=x2_cat.shape[2:])
        mff3_out = self.mff3([x1_cat, x2_cat, x3_cat, x3_cat], target_size=x3_cat.shape[2:])

        x = self.decoder_3(mff3_out, x_mid)
        x = self.decoder_2(mff2_out, x)
        x = self.decoder_1(mff1_out, x)
        x = self.binary_last_conv(x)
        return x, self.softmax(x)


    def ENC_4_DEC_4(self, x_in):
        # encoders
        x1_cat = self.encoder_1(x_in)
        x1 = self.max_pool(x1_cat)
        x2_cat = self.encoder_2(x1); x2 = self.max_pool(self.dropout(x2_cat))
        x3_cat = self.encoder_3(x2); x3 = self.max_pool(x3_cat)
        x4_cat = self.encoder_4(x3); x4 = self.max_pool(self.dropout(x4_cat))

        # bottleneck
        x_mid = self.activate(self.mid_conv_512_1024(x4))
        x_mid = self.activate(self.mid_conv_1024_1024(x_mid))
        x_mid = self.dropout(x_mid)

        # build skip connections via MFF
        mff1_out = self.mff1([x1_cat, x2_cat, x3_cat, x4_cat], target_size=x1_cat.shape[2:])
        mff2_out = self.mff2([x1_cat, x2_cat, x3_cat, x4_cat], target_size=x2_cat.shape[2:])
        mff3_out = self.mff3([x1_cat, x2_cat, x3_cat, x4_cat], target_size=x3_cat.shape[2:])
        mff4_out = self.mff4([x1_cat, x2_cat, x3_cat, x4_cat], target_size=x4_cat.shape[2:])

        # decoders with MFF outputs instead of raw encoder features
        x = self.decoder_4(mff4_out, x_mid)
        x = self.decoder_3(mff3_out, x)
        x = self.decoder_2(mff2_out, x)
        x = self.decoder_1(mff1_out, x)

        x = self.binary_last_conv(x)
        return x, self.softmax(x)


