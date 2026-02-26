from transformers import SegformerForSemanticSegmentation, SegformerModel, SegformerDecodeHead
from transformers.models.segformer.configuration_segformer import SegformerConfig
import torch
import torch.nn as nn
from base import BaseModel
from torch.optim import *
from torchvision import models

class CustomSegformer(nn.Module):
    def __init__(self, input_channels, num_classes, base_model = 'nvidia/mit-b3'):
        super().__init__()
        config = SegformerConfig.from_pretrained(base_model)
        config.num_labels = num_classes
        config.num_channels = input_channels
        self.encoder = SegformerModel(config)
        self.decoder = SegformerDecodeHead(config)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_in):
        outputs = self.encoder(x_in,
                               output_attentions = False,
                               output_hidden_states = True,
                               return_dict = True)
        
        logits = self.decoder(outputs.hidden_states)
        x = nn.functional.interpolate(logits, size=(x_in.shape[2], x_in.shape[3]), mode='bilinear', align_corners=False)
        return x, self.softmax(x)