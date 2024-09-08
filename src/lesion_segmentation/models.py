import segmentation_models_pytorch as smp
import torch.nn as nn
from torch.cuda.amp import autocast

class ISIC_2018_Seg_Model(nn.Module):
    def __init__(self, encoder_name, encoder_weights, decoder_name):
        super(ISIC_2018_Seg_Model, self).__init__()
        if decoder_name == 'UnetPlusPlus':
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1,
            )
        elif decoder_name == 'FPN':
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1,
            )
        else:
            raise ValueError("decoder_name")

    @autocast()
    def forward(self, x):
        return self.model(x)