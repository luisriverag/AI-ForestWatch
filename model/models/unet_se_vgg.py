"""
UNetSE — VGG11-backbone UNet with Squeeze-and-Excitation blocks.
Supports multiple encoder-decoder topologies (ENC_1_DEC_1 through ENC_4_DEC_4).
"""

import torch
import torch.nn as nn
from torchvision import models

from .se_blocks import (
    BaseModel, _make_se,
    UNetSE_down_block, UNetSE_up_block,
)


class UNetSE(BaseModel):
    def __init__(self, topology, input_channels, num_classes, se_reduction=16, se_flags=None):
        super(UNetSE, self).__init__()

        # NEW: SE flags
        if se_flags is None:
            se_flags = {
                "input": True,
                "encoder": True,
                "decoder": True,
                "bottleneck": False  # not used in your code yet
            }
        self.se_flags = se_flags

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
        self.dropout = nn.Dropout2d(0.5)
        self.activate = nn.ReLU(inplace=True)

        # NEW: SE on input
        self.se_input = _make_se(input_channels, max(2, input_channels // 2)) if self.se_flags["input"] else None

        # Encoders with SE
        self.encoder_1 = UNetSE_down_block(input_channels, 64,
                                           se_reduction=se_reduction,
                                           use_se=self.se_flags["encoder"])
        self.encoder_2 = UNetSE_down_block(64, 128,
                                           conv_1=pretrained_layers[3],
                                           se_reduction=se_reduction,
                                           use_se=self.se_flags["encoder"])
        self.encoder_3 = UNetSE_down_block(128, 256,
                                           conv_1=pretrained_layers[6],
                                           conv_2=pretrained_layers[8],
                                           se_reduction=se_reduction,
                                           use_se=self.se_flags["encoder"])
        self.encoder_4 = UNetSE_down_block(256, 512,
                                           conv_1=pretrained_layers[11],
                                           conv_2=pretrained_layers[13],
                                           se_reduction=se_reduction,
                                           use_se=self.se_flags["encoder"])

        self.mid_conv_64_64_a = nn.Conv2d(64, 64, 3, padding=1)
        self.mid_conv_64_64_b = nn.Conv2d(64, 64, 3, padding=1)
        self.mid_conv_128_128_a = nn.Conv2d(128, 128, 3, padding=1)
        self.mid_conv_128_128_b = nn.Conv2d(128, 128, 3, padding=1)
        self.mid_conv_256_256_a = nn.Conv2d(256, 256, 3, padding=1)
        self.mid_conv_256_256_b = nn.Conv2d(256, 256, 3, padding=1)
        self.mid_conv_512_1024 = nn.Conv2d(512, 1024, 3, padding=1)
        self.mid_conv_1024_1024 = nn.Conv2d(1024, 1024, 3, padding=1)

        # Decoders with SE
        self.decoder_4 = UNetSE_up_block(prev_channel=self.encoder_4.output_channels,
                                         input_channel=self.mid_conv_1024_1024.out_channels,
                                         output_channel=256,
                                         se_reduction=se_reduction,
                                         use_se=self.se_flags["decoder"])
        self.decoder_3 = UNetSE_up_block(prev_channel=self.encoder_3.output_channels,
                                         input_channel=self.decoder_4.output_channels,
                                         output_channel=128,
                                         se_reduction=se_reduction,
                                         use_se=self.se_flags["decoder"])
        self.decoder_2 = UNetSE_up_block(prev_channel=self.encoder_2.output_channels,
                                         input_channel=self.decoder_3.output_channels,
                                         output_channel=64,
                                         se_reduction=se_reduction,
                                         use_se=self.se_flags["decoder"])
        self.decoder_1 = UNetSE_up_block(prev_channel=self.encoder_1.output_channels,
                                         input_channel=self.decoder_2.output_channels,
                                         output_channel=64,
                                         se_reduction=se_reduction,
                                         use_se=self.se_flags["decoder"])

        self.binary_last_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.forward = self.topologies[topology]
        print('\n\n' + "#" * 100)
        print("(LOG): The following Model Topology will be Utilized: {}".format(
            self.forward.__name__))
        print("#" * 100 + '\n\n')

    def _maybe_se_input(self, x_in):
        if self.se_input is not None:
            return self.se_input(x_in)
        return x_in

    def ENC_1_DEC_1(self, x_in):
        x_in = self._maybe_se_input(x_in)  # NEW
        x1_cat = self.encoder_1(x_in)
        x1_cat_1 = self.dropout(x1_cat)
        x1 = self.max_pool(x1_cat_1)

        x_mid = self.mid_conv_64_64_a(x1); x_mid = self.activate(x_mid)
        x_mid = self.mid_conv_64_64_b(x_mid); x_mid = self.activate(x_mid)
        x_mid = self.dropout(x_mid)

        x = self.decoder_1(x1_cat, x_mid)
        x = self.binary_last_conv(x)
        return x, self.softmax(x)

    def ENC_2_DEC_2(self, x_in):
        x_in = self._maybe_se_input(x_in)  # NEW
        x1_cat = self.encoder_1(x_in)
        x1 = self.max_pool(x1_cat)

        x2_cat = self.encoder_2(x1)
        x2_cat_1 = self.dropout(x2_cat)
        x2 = self.max_pool(x2_cat_1)

        x_mid = self.mid_conv_128_128_a(x2); x_mid = self.activate(x_mid)
        x_mid = self.mid_conv_128_128_b(x_mid); x_mid = self.activate(x_mid)
        x_mid = self.dropout(x_mid)

        x = self.decoder_2(x2_cat, x_mid)
        x = self.decoder_1(x1_cat, x)
        x = self.binary_last_conv(x)
        return x, self.softmax(x)

    def ENC_3_DEC_3(self, x_in):
        x_in = self._maybe_se_input(x_in)  # NEW
        x1_cat = self.encoder_1(x_in)
        x1 = self.max_pool(x1_cat)

        x2_cat = self.encoder_2(x1)
        x2_cat_1 = self.dropout(x2_cat)
        x2 = self.max_pool(x2_cat_1)

        x3_cat = self.encoder_3(x2)
        x3 = self.max_pool(x3_cat)

        x_mid = self.mid_conv_256_256_a(x3); x_mid = self.activate(x_mid)
        x_mid = self.mid_conv_256_256_b(x_mid); x_mid = self.activate(x_mid)
        x_mid = self.dropout(x_mid)

        x = self.decoder_3(x3_cat, x_mid)
        x = self.decoder_2(x2_cat, x)
        x = self.decoder_1(x1_cat, x)
        x = self.binary_last_conv(x)
        return x, self.softmax(x)

    def ENC_4_DEC_4(self, x_in):
        x_in = self._maybe_se_input(x_in)  # NEW
        x1_cat = self.encoder_1(x_in)
        x1 = self.max_pool(x1_cat)

        x2_cat = self.encoder_2(x1)
        x2_cat_1 = self.dropout(x2_cat)
        x2 = self.max_pool(x2_cat_1)

        x3_cat = self.encoder_3(x2)
        x3 = self.max_pool(x3_cat)

        x4_cat = self.encoder_4(x3)
        x4_cat_1 = self.dropout(x4_cat)
        x4 = self.max_pool(x4_cat_1)

        x_mid = self.mid_conv_512_1024(x4); x_mid = self.activate(x_mid)
        x_mid = self.mid_conv_1024_1024(x_mid); x_mid = self.activate(x_mid)
        x_mid = self.dropout(x_mid)

        x = self.decoder_4(x4_cat, x_mid)
        x = self.decoder_3(x3_cat, x)
        x = self.decoder_2(x2_cat, x)
        x = self.decoder_1(x1_cat, x)
        x = self.binary_last_conv(x)
        return x, self.softmax(x)
