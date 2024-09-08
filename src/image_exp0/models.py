import torch
import torch.nn as nn
import timm
from torch.cuda.amp import autocast

class ISIC_Model(nn.Module):
    def __init__(self, model_name, pretrained, num_classes):
        super(ISIC_Model, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    @autocast()
    def forward(self, x):
        return self.backbone(x)