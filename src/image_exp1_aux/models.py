import torch.nn as nn
from timm.models.layers import BatchNormAct2d, SelectAdaptivePool2d
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import Optional, List

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init

class IsicSMPEffnetUnetModel(nn.Module):
    def __init__(
        self,
        encoder_name = "timm-efficientnet-b0",
        encoder_weights = 'noisy-student',
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_features: int = 1280,
        classes: int = 4,
        test_mode=False,
    ):
        super(IsicSMPEffnetUnetModel, self).__init__()
        self.in_features = in_features
        self.test_mode = test_mode

        self.encoder = get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )

        self.conv_head = self.encoder.conv_head
        self.bn2 = self.encoder.bn2
        self.global_pool = self.encoder.global_pool

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            activation=None,
            kernel_size=3,
        )
        self.fc = nn.Linear(in_features, 1024, bias=True)
        self.cls_head = nn.Linear(1024, classes, bias=True)

        init.initialize_head(self.fc)
        init.initialize_head(self.cls_head)
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
    
    @autocast()
    def forward(self, x):
        x = self.encoder(x)
        y_cls = self.conv_head(x[-1])
        y_cls = self.bn2(y_cls)
        y_cls = self.global_pool(y_cls)
        y_cls = y_cls.view(-1, self.in_features)
        y_cls = self.fc(y_cls)
        y_cls = F.relu(y_cls)
        y_cls = F.dropout(y_cls, p=0.5, training=self.training)
        y_cls = self.cls_head(y_cls)
        
        if self.test_mode:
            return y_cls
        else:
            y_seg = self.decoder(*x)
            y_seg = self.segmentation_head(y_seg)
            return y_cls, y_seg

class IsicSMPMitFPNModel(nn.Module):
    def __init__(
        self,
        encoder_name = "mit_b5",
        encoder_weights = 'imagenet',
        classes: int = 4,
        test_mode=False,
    ):
        super(IsicSMPMitFPNModel, self).__init__()
        self.in_features = 1280
        self.test_mode = test_mode

        self.encoder = get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )
        if encoder_name == "mit_b5" or encoder_name == 'mit_b3':
            self.conv_head = nn.Conv2d(512, self.in_features, 1, 1, bias=False)
        elif encoder_name == "mit_b0":
            self.conv_head = nn.Conv2d(256, self.in_features, 1, 1, bias=False)
        else:
            raise ValueError()
        self.bn2 = BatchNormAct2d(num_features=self.in_features)
        self.global_pool = SelectAdaptivePool2d()

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy="add",
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=1,
            kernel_size=1,
            upsampling=4,
        )
        self.fc = nn.Linear(self.in_features, 1024, bias=True)
        self.cls_head = nn.Linear(1024, classes, bias=True)

        init.initialize_head(self.fc)
        init.initialize_head(self.cls_head)
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
    
    @autocast()
    def forward(self, x):
        x = self.encoder(x)

        y_cls = self.conv_head(x[-1])
        y_cls = self.bn2(y_cls)
        y_cls = self.global_pool(y_cls)
        y_cls = y_cls.view(-1, self.in_features)
        y_cls = self.fc(y_cls)
        y_cls = F.relu(y_cls)
        y_cls = F.dropout(y_cls, p=0.5, training=self.training)
        y_cls = self.cls_head(y_cls)

        if self.test_mode:
            return y_cls
        else:
            y_seg = self.decoder(*x)
            y_seg = self.segmentation_head(y_seg)
            return y_cls, y_seg

class IsicAuxModel(nn.Module):
    def __init__(self, encoder_name="timm-efficientnet-b0", encoder_weights = 'noisy-student', in_features=1280, classes=10, img_size=256, test_mode=False):
        super(IsicAuxModel, self).__init__()
        if "timm-efficientnet" in encoder_name:
            self.model = IsicSMPEffnetUnetModel(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_features=in_features,
                classes=classes,
                test_mode=test_mode)
        elif "mit_b" in encoder_name:
            self.model = IsicSMPMitFPNModel(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=classes,
                test_mode=test_mode)
        else:
            raise ValueError()
    @autocast()
    def forward(self, x):
        return self.model(x)
