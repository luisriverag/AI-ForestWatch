import torch
import torch.nn as nn
from torch.optim import *
from torchvision import models
from torchvision.ops import SqueezeExcitation  # NEW
import timm

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

class UNetSE_resnet(BaseModel):
    def __init__(self, num_classes, input_channels=3, se_reduction=16, se_flags=None):
        super().__init__()

        if se_flags is None:
            se_flags = {"input": True, "encoder": True, "decoder": True, "bottleneck": False}
        self.se_flags = se_flags

        # # ImageNet pretrained ResNet50 backbone
        # resnet = models.resnet50(pretrained=True)

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

        # Create the new first layer for 18 channels (don't load pretrained weights for this)
        self.layer0_conv = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize with Xavier/He initialization
        nn.init.xavier_uniform_(self.layer0_conv.weight)

        # Load pretrained weights for the rest of ResNet (skip conv1)
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k != 'conv1.weight' and k != 'conv1.bias'}
        resnet.load_state_dict(pretrained_dict, strict=False)

        # Use the pretrained batch norm and activation from the checkpoint
        self.layer0_bn = resnet.bn1
        self.layer0_relu = resnet.relu

        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

        # Freeze all encoder layers except layer0
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False
        # self.layer0 is left trainable (default)

        # Optional SE on input
        self.se_input = _make_se(input_channels, max(2, input_channels // 2)) if self.se_flags["input"] else None

        # Decoder path
        self.decoder4 = UNetSE_up_block(prev_channel=1024, input_channel=2048,
                                        output_channel=512, se_reduction=se_reduction,
                                        use_se=self.se_flags["decoder"])
        self.decoder3 = UNetSE_up_block(prev_channel=512, input_channel=512,
                                        output_channel=256, se_reduction=se_reduction,
                                        use_se=self.se_flags["decoder"])
        self.decoder2 = UNetSE_up_block(prev_channel=256, input_channel=256,
                                        output_channel=128, se_reduction=se_reduction,
                                        use_se=self.se_flags["decoder"])
        self.decoder1 = UNetSE_up_block(prev_channel=64, input_channel=128,
                                        output_channel=64, se_reduction=se_reduction,
                                        use_se=self.se_flags["decoder"])
                                        
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
        x1 = self.layer1(self.maxpool(x0)) # 256
        x2 = self.layer2(x1)               # 512
        x3 = self.layer3(x2)               # 1024
        x4 = self.layer4(x3)               # 2048

        # Decoder path with skip connections
        x = self.decoder4(x3, x4)
        x = self.decoder3(x2, x)
        x = self.decoder2(x1, x)
        x = self.decoder1(x0, x)
        x = self.decoder0(x)

        x = self.final_conv(x)
        return x, self.softmax(x)



if __name__ == "__main__":

    print("\n" + "#" * 100)
    print("Testing UNetSE_resnet architecture")
    print("#" * 100 + "\n")

    # Instantiate your UNetSE_resnet
    model = UNetSE_resnet(num_classes=2, input_channels=3)
    print(model)

    # Quick forward pass test with dummy input
    x = torch.randn(1, 3, 128, 128)   # batch=1, RGB image, 224x224
    y, y_soft = model(x)
    print("\nOutput tensor shape (logits):", y.shape)
    print("Output tensor shape (softmax):", y_soft.shape)
