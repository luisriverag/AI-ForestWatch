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

class UNet3PlusMFFSE(BaseModel):
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
        super(UNet3PlusMFFSE, self).__init__()

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

class UNet3PlusSE(BaseModel):
    """
    UNet 3+ variant with Squeeze-and-Excitation (SE) using UNetSE_down_block encoders.

    - Encoder: UNetSE_down_block (VGG11-seeded, with optional SE)
    - Bottleneck: same as UNet3Plus, with optional SE on the bottleneck features
    - Decoder: full-scale feature aggregation at level-1 resolution, with optional SE on fused features
    - Output: (logits, softmax(logits)) to match existing training loop
    """

    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 se_reduction: int = 16,
                 se_flags: dict = None,
                 dropout: float = 0.1):
        super(UNet3PlusSE, self).__init__()

        self.dropout_p = dropout
        self.se_reduction = se_reduction

        # SE flags, similar to UNetSE but adapted for 3+ aggregation
        # "fusion" controls SE on the concatenated multi-scale feature map
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

        # basic ops
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(self.dropout_p)
        self.softmax = nn.Softmax(dim=1)
        self.activate = nn.ReLU(inplace=True)

        # SE on raw input (optional, like in UNetSE)
        self.se_input = _make_se(
            input_channels,
            max(2, input_channels // 2)
        ) if self.se_flags.get("input", False) else None

        # -----------------------
        # Encoder: UNetSE_down_block
        # -----------------------
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

        # # -----------------------
        # # Bottleneck (same as UNet3Plus)
        # # -----------------------
        # self.mid_conv_512_1024 = nn.Conv2d(512, 1024, 3, padding=1)
        # self.mid_conv_1024_1024 = nn.Conv2d(1024, 1024, 3, padding=1)
        
        # Bottom (1/16)
        self.bottom = ConvBNReLU(512, 1024, k=3, s=1, p=1, use_se=True, se_reduction=self.se_reduction)

        # Optional SE on bottleneck features
        self.se_bottleneck = _make_se(
            1024, se_reduction
        ) if self.se_flags.get("bottleneck", True) else None

        # -----------------------
        # Project all encoder scales (and bottom) to constant width (C) for fusion
        # -----------------------
        agg_channels = 64
        self.proj1 = nn.Conv2d(64, agg_channels, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(128, agg_channels, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(256, agg_channels, kernel_size=1, bias=False)
        self.proj4 = nn.Conv2d(512, agg_channels, kernel_size=1, bias=False)
        self.proj5 = nn.Conv2d(1024, agg_channels, kernel_size=1, bias=False)

        # Decoder fusion blocks: concat 5 tensors (→ 5*C), then refine
        self.decoder_4 = ConvBNReLU(5*64, 512)
        self.decoder_3 = ConvBNReLU(5*64, 256)
        self.decoder_2 = ConvBNReLU(5*64, 128)
        self.decoder_1 = ConvBNReLU(5*64, 64)

        # SE on the concatenated multi-scale features (optional)
        fusion_in = agg_channels * 5
        self.se_fusion = _make_se(
            fusion_in, se_reduction
        ) if self.se_flags.get("fusion", True) else None

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

    def _maybe_se_input(self, x_in):
        if self.se_input is not None:
            return self.se_input(x_in)
        return x_in

    def _maybe_se_bottleneck(self, x):
        if self.se_bottleneck is not None:
            return self.se_bottleneck(x)
        return x

    def _maybe_se_fusion(self, x):
        if self.se_fusion is not None:
            return self.se_fusion(x)
        return x

    def forward(self, x_in):
        # Optional SE on input
        x_in = self._maybe_se_input(x_in)

        # ---------------- Encoder ----------------
        e1 = self.encoder_1(x_in)                              # [B,  C, H,   W]
        e2 = self.encoder_2(self.max_pool(self.dropout(e1)))   # [B, 2C, H/2, W/2]
        e3 = self.encoder_3(self.max_pool(self.dropout(e2)))   # [B, 4C, H/4, W/4]
        e4 = self.encoder_4(self.max_pool(self.dropout(e3)))   # [B, 8C, H/8, W/8]
        btm = self.bottom(self.max_pool(self.dropout(e4)))# [B,16C, H/16, W/16]

        # Projections to C channels
        p1 = self.proj1(e1)
        p2 = self.proj2(e2)
        p3 = self.proj3(e3)
        p4 = self.proj4(e4)
        p5 = self.proj5(btm)

        # ---------------- D4 @ 1/8 ----------------
        t4 = e4
        f4 = torch.cat([
            self._resize_to(p1, t4),
            self._resize_to(p2, t4),
            self._resize_to(p3, t4),
            p4,                        # already 1/8
            self._resize_to(p5, t4),
        ], dim=1)                      # 5*C
        d4 = self.decoder_4(f4)             # [B, 8C, 1/8]

        # ---------------- D3 @ 1/4 ----------------
        t3 = e3
        d4_c = self._resize_to(self.proj_d4(d4), t3)  # project d4->C then resize
        f3 = torch.cat([
            self._resize_to(p1, t3),
            self._resize_to(p2, t3),
            p3,
            d4_c,
            self._resize_to(p5, t3),
        ], dim=1)                      # 5*C
        d3 = self.decoder_3(f3)             # [B, 4C, 1/4]

        # ---------------- D2 @ 1/2 ----------------
        t2 = e2
        d3_c = self._resize_to(self.proj_d3(d3), t2)  # project d3->C then resize
        f2 = torch.cat([
            self._resize_to(p1, t2),
            p2,
            d3_c,
            self._resize_to(p4, t2),
            self._resize_to(p5, t2),
        ], dim=1)                      # 5*C
        d2 = self.decoder_2(f2)             # [B, 2C, 1/2]

         # ---------------- D1 @ 1/1 ----------------
        t1 = e1
        d2_c = self._resize_to(self.proj_d2(d2), t1)  # project d2->C then resize
        f1 = torch.cat([
            p1,
            d2_c,
            self._resize_to(p3, t1),
            self._resize_to(p4, t1),
            self._resize_to(p5, t1),
        ], dim=1)                      # 5*C
        d1 = self.decoder_1(f1)             # [B,  C, 1/1]

        logits = self.head(d1)         # [B, num_classes, H, W]
        return logits, self.softmax(logits)


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