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

class UNet3PlusSE(BaseModel):
    """
    UNet 3+ variant with optional Multi-Feature Fusion (MFF) and SE blocks.
    Can toggle MFF on/off. If MFF is off, behaves like UNet3PlusSE (standard aggregation).
    """

    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 se_reduction: int = 16,
                 se_flags: dict = None,
                 dropout: float = 0.1,
                 use_mff: bool = True):
        super(UNet3PlusSE, self).__init__()

        self.dropout_p = dropout
        self.se_reduction = se_reduction
        self.use_mff = use_mff

        if se_flags is None:
            se_flags = {
                "input": False,
                "encoder": True,
                "fusion": False,
                "bottleneck": False
            }
        self.se_flags = se_flags

        vgg_trained = models.vgg11(pretrained=True)
        pretrained_layers = list(vgg_trained.features)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(self.dropout_p)
        self.softmax = nn.Softmax(dim=1)
        self.activate = nn.ReLU(inplace=True)

        self.se_input = _make_se(
            input_channels,
            max(2, input_channels // 2)
        ) if self.se_flags.get("input", False) else None

        # ---------------- Encoder ----------------
        self.encoder_1 = UNetSE_down_block(
            input_channel=input_channels,
            output_channel=64,
            se_reduction=se_reduction,
            use_se=self.se_flags.get("encoder", True)
        )
        self.encoder_2 = UNetSE_down_block(
            input_channel=64,
            output_channel=128,
            conv_1=pretrained_layers[3],
            se_reduction=se_reduction,
            use_se=self.se_flags.get("encoder", True)
        )
        self.encoder_3 = UNetSE_down_block(
            input_channel=128,
            output_channel=256,
            conv_1=pretrained_layers[6],
            conv_2=pretrained_layers[8],
            se_reduction=se_reduction,
            use_se=self.se_flags.get("encoder", True)
        )
        self.encoder_4 = UNetSE_down_block(
            input_channel=256,
            output_channel=512,
            conv_1=pretrained_layers[11],
            conv_2=pretrained_layers[13],
            se_reduction=se_reduction,
            use_se=self.se_flags.get("encoder", True)
        )

        self.bottom = ConvBNReLU(512, 1024, k=3, s=1, p=1, use_se=True, se_reduction=self.se_reduction)

        self.se_bottleneck = _make_se(
            1024, se_reduction
        ) if self.se_flags.get("bottleneck", True) else None

        # ---------------- Aggregation / MFF ----------------
        agg_channels = 64
        # Channel counts from encoders: 64, 128, 256, 512, 1024 (bottom)
        
        if self.use_mff:
            # MFF Blocks replace separate projections and concatenation
            # One MFF block per decoder level
            # Decoder 4 inputs: e1, e2, e3, e4, bottom
            self.mff4 = MFFBlock([64, 128, 256, 512, 1024], agg_channels)
            self.mff3 = MFFBlock([64, 128, 256, 512, 1024], agg_channels)
            self.mff2 = MFFBlock([64, 128, 256, 512, 1024], agg_channels)
            self.mff1 = MFFBlock([64, 128, 256, 512, 1024], agg_channels)
        else:
            # Standard independent projections
            self.proj1 = nn.Conv2d(64, agg_channels, kernel_size=1, bias=False)
            self.proj2 = nn.Conv2d(128, agg_channels, kernel_size=1, bias=False)
            self.proj3 = nn.Conv2d(256, agg_channels, kernel_size=1, bias=False)
            self.proj4 = nn.Conv2d(512, agg_channels, kernel_size=1, bias=False)
            self.proj5 = nn.Conv2d(1024, agg_channels, kernel_size=1, bias=False)

        # ---------------- Decoders ----------------
        # If MFF is used, input to decoder is the fused output (agg_channels)
        # If MFF is NOT used, input is concat of 5 projections (5 * agg_channels)
        dec_in_channels = agg_channels if self.use_mff else (agg_channels * 5)

        self.decoder_4 = ConvBNReLU(dec_in_channels, 512)
        self.decoder_3 = ConvBNReLU(dec_in_channels, 256)
        self.decoder_2 = ConvBNReLU(dec_in_channels, 128)
        self.decoder_1 = ConvBNReLU(dec_in_channels, 64)

        # SE on fusion (Only makes sense for standard concatenation, MFF has internal fusion)
        # We'll allow it if configured, but for MFF it applies to the MFF output
        self.se_fusion = _make_se(
            dec_in_channels, se_reduction
        ) if self.se_flags.get("fusion", True) else None

        if not self.use_mff:
            self.proj_d4 = nn.Conv2d(512, 64, kernel_size=1, bias=False)
            self.proj_d3 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
            self.proj_d2 = nn.Conv2d(128, 64, kernel_size=1, bias=False)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def _resize_to(self, x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    
    def _act_bn(self, bn_layer, x):
        return self.activate(bn_layer(x))

    def _maybe_se_input(self, x): return self.se_input(x) if self.se_input else x
    def _maybe_se_bottleneck(self, x): return self.se_bottleneck(x) if self.se_bottleneck else x
    def _maybe_se_fusion(self, x): return self.se_fusion(x) if self.se_fusion else x

    def forward(self, x_in):
        x_in = self._maybe_se_input(x_in)
        
        # Encoders
        e1 = self.encoder_1(x_in)
        e2 = self.encoder_2(self.max_pool(self.dropout(e1)))
        e3 = self.encoder_3(self.max_pool(self.dropout(e2)))
        e4 = self.encoder_4(self.max_pool(self.dropout(e3)))
        btm = self.bottom(self.max_pool(self.dropout(e4)))
        btm = self._maybe_se_bottleneck(btm)

        if self.use_mff:
            return self._forward_mff(e1, e2, e3, e4, btm)
        else:
            return self._forward_std(e1, e2, e3, e4, btm)

    def _forward_mff(self, e1, e2, e3, e4, btm):
        # D4
        # MFF inputs: [E1, E2, E3, E4, Btm]. Target size: E4 (1/8)
        f4 = self.mff4([e1, e2, e3, e4, btm], target_size=e4.shape[-2:])
        f4 = self._maybe_se_fusion(f4)
        d4 = self.decoder_4(f4)

        # D3
        # MFF inputs: [E1, E2, E3, D4, Btm]. Target size: E3 (1/4)
        f3 = self.mff3([e1, e2, e3, d4, btm], target_size=e3.shape[-2:])
        f3 = self._maybe_se_fusion(f3)
        d3 = self.decoder_3(f3)

        # D2
        # MFF inputs: [E1, E2, D3, E4, Btm] (replacing Slot 3 with D3?)
        # Standard: [p1, p2, d3_c, p4, p5]. Slot 3 is D3.
        f2 = self.mff2([e1, e2, d3, e4, btm], target_size=e2.shape[-2:])
        f2 = self._maybe_se_fusion(f2)
        d2 = self.decoder_2(f2)

        # D1
        # MFF inputs: [E1, D2, E3, E4, Btm] (replacing Slot 2 with D2?)
        # Standard: [p1, d2_c, p3, p4, p5]. Slot 2 is D2.
        f1 = self.mff1([e1, d2, e3, e4, btm], target_size=e1.shape[-2:])
        f1 = self._maybe_se_fusion(f1)
        d1 = self.decoder_1(f1)

        logits = self.head(d1)
        return logits, self.softmax(logits)

    def _forward_std(self, e1, e2, e3, e4, btm):
        # Projections
        p1 = self.proj1(e1)
        p2 = self.proj2(e2)
        p3 = self.proj3(e3)
        p4 = self.proj4(e4)
        p5 = self.proj5(btm)

        # D4
        t4 = e4
        f4 = torch.cat([
            self._resize_to(p1, t4),
            self._resize_to(p2, t4),
            self._resize_to(p3, t4),
            p4,
            self._resize_to(p5, t4),
        ], dim=1)
        f4 = self._maybe_se_fusion(f4)
        d4 = self.decoder_4(f4)

        # D3
        t3 = e3
        d4_c = self._resize_to(self.proj_d4(d4), t3)
        f3 = torch.cat([
            self._resize_to(p1, t3),
            self._resize_to(p2, t3),
            p3,
            d4_c,
            self._resize_to(p5, t3),
        ], dim=1)
        f3 = self._maybe_se_fusion(f3)
        d3 = self.decoder_3(f3)

        # D2
        t2 = e2
        d3_c = self._resize_to(self.proj_d3(d3), t2)
        f2 = torch.cat([
            self._resize_to(p1, t2),
            p2,
            d3_c,
            self._resize_to(p4, t2),
            self._resize_to(p5, t2),
        ], dim=1)
        f2 = self._maybe_se_fusion(f2)
        d2 = self.decoder_2(f2)

        # D1
        t1 = e1
        d2_c = self._resize_to(self.proj_d2(d2), t1)
        f1 = torch.cat([
            p1,
            d2_c,
            self._resize_to(p3, t1),
            self._resize_to(p4, t1),
            self._resize_to(p5, t1),
        ], dim=1)
        f1 = self._maybe_se_fusion(f1)
        d1 = self.decoder_1(f1)

        logits = self.head(d1)
        return logits, self.softmax(logits)

# class UNet3PlusSE(BaseModel):
#     """
#     UNet 3+ variant with Squeeze-and-Excitation (SE) using UNetSE_down_block encoders.

#     - Encoder: UNetSE_down_block (VGG11-seeded, with optional SE)
#     - Bottleneck: same as UNet3Plus, with optional SE on the bottleneck features
#     - Decoder: full-scale feature aggregation at level-1 resolution, with optional SE on fused features
#     - Output: (logits, softmax(logits)) to match existing training loop
#     """

#     def __init__(self,
#                  input_channels: int,
#                  num_classes: int,
#                  se_reduction: int = 16,
#                  se_flags: dict = None,
#                  dropout: float = 0.1):
#         super(UNet3PlusSE, self).__init__()

#         self.dropout_p = dropout
#         self.se_reduction = se_reduction

#         # SE flags, similar to UNetSE but adapted for 3+ aggregation
#         # "fusion" controls SE on the concatenated multi-scale feature map
#         if se_flags is None:
#             se_flags = {
#                 "input": False,
#                 "encoder": True,
#                 "fusion": False,
#                 "bottleneck": False
#             }
        
#         self.se_flags = se_flags

#         vgg_trained = models.vgg11(pretrained=True)
#         pretrained_layers = list(vgg_trained.features)

#         # basic ops
#         self.max_pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout2d(self.dropout_p)
#         self.softmax = nn.Softmax(dim=1)
#         self.activate = nn.ReLU(inplace=True)

#         # SE on raw input (optional, like in UNetSE)
#         self.se_input = _make_se(
#             input_channels,
#             max(2, input_channels // 2)
#         ) if self.se_flags.get("input", False) else None

#         # -----------------------
#         # Encoder: UNetSE_down_block
#         # -----------------------
#         self.encoder_1 = UNetSE_down_block(
#             input_channel=input_channels,
#             output_channel=64,
#             se_reduction=se_reduction,
#             use_se=self.se_flags.get("encoder", True)
#         )

#         self.encoder_2 = UNetSE_down_block(
#             input_channel=64,
#             output_channel=128,
#             conv_1=pretrained_layers[3],
#             se_reduction=se_reduction,
#             use_se=self.se_flags.get("encoder", True)
#         )

#         self.encoder_3 = UNetSE_down_block(
#             input_channel=128,
#             output_channel=256,
#             conv_1=pretrained_layers[6],
#             conv_2=pretrained_layers[8],
#             se_reduction=se_reduction,
#             use_se=self.se_flags.get("encoder", True)
#         )

#         self.encoder_4 = UNetSE_down_block(
#             input_channel=256,
#             output_channel=512,
#             conv_1=pretrained_layers[11],
#             conv_2=pretrained_layers[13],
#             se_reduction=se_reduction,
#             use_se=self.se_flags.get("encoder", True)
#         )

#         # # -----------------------
#         # # Bottleneck (same as UNet3Plus)
#         # # -----------------------
#         # self.mid_conv_512_1024 = nn.Conv2d(512, 1024, 3, padding=1)
#         # self.mid_conv_1024_1024 = nn.Conv2d(1024, 1024, 3, padding=1)
        
#         # Bottom (1/16)
#         self.bottom = ConvBNReLU(512, 1024, k=3, s=1, p=1, use_se=True, se_reduction=self.se_reduction)

#         # Optional SE on bottleneck features
#         self.se_bottleneck = _make_se(
#             1024, se_reduction
#         ) if self.se_flags.get("bottleneck", True) else None

#         # -----------------------
#         # Project all encoder scales (and bottom) to constant width (C) for fusion
#         # -----------------------
#         agg_channels = 64
#         self.proj1 = nn.Conv2d(64, agg_channels, kernel_size=1, bias=False)
#         self.proj2 = nn.Conv2d(128, agg_channels, kernel_size=1, bias=False)
#         self.proj3 = nn.Conv2d(256, agg_channels, kernel_size=1, bias=False)
#         self.proj4 = nn.Conv2d(512, agg_channels, kernel_size=1, bias=False)
#         self.proj5 = nn.Conv2d(1024, agg_channels, kernel_size=1, bias=False)

#         # Decoder fusion blocks: concat 5 tensors (→ 5*C), then refine
#         self.decoder_4 = ConvBNReLU(5*64, 512)
#         self.decoder_3 = ConvBNReLU(5*64, 256)
#         self.decoder_2 = ConvBNReLU(5*64, 128)
#         self.decoder_1 = ConvBNReLU(5*64, 64)

#         # SE on the concatenated multi-scale features (optional)
#         fusion_in = agg_channels * 5
#         self.se_fusion = _make_se(
#             fusion_in, se_reduction
#         ) if self.se_flags.get("fusion", True) else None

#         self.proj_d4 = nn.Conv2d(512, 64, kernel_size=1, bias=False)
#         self.proj_d3 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
#         self.proj_d2 = nn.Conv2d(128, 64, kernel_size=1, bias=False)

#         self.head = nn.Conv2d(64, num_classes, kernel_size=1)

#     def _resize_to(self, x, ref):
#         if x.shape[-2:] == ref.shape[-2:]:
#             return x
#         return nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

#     def _act_bn(self, bn_layer, x):
#         return self.activate(bn_layer(x))

#     def _maybe_se_input(self, x_in):
#         if self.se_input is not None:
#             return self.se_input(x_in)
#         return x_in

#     def _maybe_se_bottleneck(self, x):
#         if self.se_bottleneck is not None:
#             return self.se_bottleneck(x)
#         return x

#     def _maybe_se_fusion(self, x):
#         if self.se_fusion is not None:
#             return self.se_fusion(x)
#         return x

#     def forward(self, x_in):
#         # Optional SE on input
#         x_in = self._maybe_se_input(x_in)

#         # ---------------- Encoder ----------------
#         e1 = self.encoder_1(x_in)                              # [B,  C, H,   W]
#         e2 = self.encoder_2(self.max_pool(self.dropout(e1)))   # [B, 2C, H/2, W/2]
#         e3 = self.encoder_3(self.max_pool(self.dropout(e2)))   # [B, 4C, H/4, W/4]
#         e4 = self.encoder_4(self.max_pool(self.dropout(e3)))   # [B, 8C, H/8, W/8]
#         btm = self.bottom(self.max_pool(self.dropout(e4)))# [B,16C, H/16, W/16]

#         # Projections to C channels
#         p1 = self.proj1(e1)
#         p2 = self.proj2(e2)
#         p3 = self.proj3(e3)
#         p4 = self.proj4(e4)
#         p5 = self.proj5(btm)

#         # ---------------- D4 @ 1/8 ----------------
#         t4 = e4
#         f4 = torch.cat([
#             self._resize_to(p1, t4),
#             self._resize_to(p2, t4),
#             self._resize_to(p3, t4),
#             p4,                        # already 1/8
#             self._resize_to(p5, t4),
#         ], dim=1)                      # 5*C
#         d4 = self.decoder_4(f4)             # [B, 8C, 1/8]

#         # ---------------- D3 @ 1/4 ----------------
#         t3 = e3
#         d4_c = self._resize_to(self.proj_d4(d4), t3)  # project d4->C then resize
#         f3 = torch.cat([
#             self._resize_to(p1, t3),
#             self._resize_to(p2, t3),
#             p3,
#             d4_c,
#             self._resize_to(p5, t3),
#         ], dim=1)                      # 5*C
#         d3 = self.decoder_3(f3)             # [B, 4C, 1/4]

#         # ---------------- D2 @ 1/2 ----------------
#         t2 = e2
#         d3_c = self._resize_to(self.proj_d3(d3), t2)  # project d3->C then resize
#         f2 = torch.cat([
#             self._resize_to(p1, t2),
#             p2,
#             d3_c,
#             self._resize_to(p4, t2),
#             self._resize_to(p5, t2),
#         ], dim=1)                      # 5*C
#         d2 = self.decoder_2(f2)             # [B, 2C, 1/2]

#          # ---------------- D1 @ 1/1 ----------------
#         t1 = e1
#         d2_c = self._resize_to(self.proj_d2(d2), t1)  # project d2->C then resize
#         f1 = torch.cat([
#             p1,
#             d2_c,
#             self._resize_to(p3, t1),
#             self._resize_to(p4, t1),
#             self._resize_to(p5, t1),
#         ], dim=1)                      # 5*C
#         d1 = self.decoder_1(f1)             # [B,  C, 1/1]

#         logits = self.head(d1)         # [B, num_classes, H, W]
#         return logits, self.softmax(logits)


if __name__ == "__main__":

    print("\n" + "#" * 100)
    print("Testing UNet3PlusSE architecture")
    print("#" * 100 + "\n")

    # Instantiate your UNetSE_resnet
    model = UNet3PlusSE(num_classes=2, input_channels=3)
    print(model)

    # Quick forward pass test with dummy input
    x = torch.randn(1, 3, 128, 128)   # batch=1, RGB image, 224x224
    y, y_soft = model(x)
    print("\nOutput tensor shape (logits):", y.shape)
    print("Output tensor shape (softmax):", y_soft.shape)