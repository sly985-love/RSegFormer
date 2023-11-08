# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.ops import carafe, CARAFEPack
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..utils import CoorAtte, RepMLPNetUnit, DUpsampling

from mmseg.models.utils import *
import attr

from IPython import embed

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.ca_c4 = CoorAtte(inp=c4_in_channels, oup=c4_in_channels)
        self.ca_c3 = CoorAtte(inp=c3_in_channels, oup=c3_in_channels)
        self.ca_c2 = CoorAtte(inp=c2_in_channels, oup=c2_in_channels)
        self.ca_c1 = CoorAtte(inp=c1_in_channels, oup=c1_in_channels)


        self.carafe_c4 = CARAFEPack(c4_in_channels,scale_factor=8,up_kernel=7, encoder_kernel=5)
        self.carafe_c3 = CARAFEPack(c3_in_channels,scale_factor=4,up_kernel=7, encoder_kernel=5)
        self.carafe_c2 = CARAFEPack(c2_in_channels,scale_factor=2,up_kernel=7, encoder_kernel=5)
        self.carafe_c1 = CARAFEPack(c1_in_channels,scale_factor=1,up_kernel=7, encoder_kernel=5)

        # self.du_c4 = DUpsampling(c4_in_channels,8, num_class=self.num_classes)
        # self.du_c3 = DUpsampling(c3_in_channels,4, num_class=self.num_classes)
        # self.du_c2 = DUpsampling(c2_in_channels,2, num_class=self.num_classes)
        # self.du_c1 = DUpsampling(c1_in_channels,1, num_class=self.num_classes)


        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            # in_channels=embedding_dim*4,
            in_channels=960,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        # self.dupsample = DUpsampling(embedding_dim, scale=8, num_class=self.num_classes)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape


        c4 = self.ca_c4(c4)
        c3 = self.ca_c3(c3)
        c2 = self.ca_c2(c2)
        c1 = self.ca_c1(c1)

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = self.carafe_c4(c4)
        # _c4 = self.du_c4(c4)
        # _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = self.carafe_c3(c3)
        # _c3 = self.du_c3(c3)
        # _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = self.carafe_c2(c2)
        # _c2 = self.du_c2(c2)
        # _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)
        _c1 = self.carafe_c1(c1)
        # _c1 = self.du_c1(c1)
        # _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


