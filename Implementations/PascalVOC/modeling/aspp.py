import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

import numpy as np


#########
CODE_SIZE = 128
SLICE_SHAPE = [16,16,3,3]
#########


class _ASPPModuleOrg(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModuleOrg, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm, CSG=None):
        super(_ASPPModule, self).__init__()
    
        self.CSG = CSG
        self.in_filters = inplanes
        self.out_filters = planes
        self.kernel_size = kernel_size
        self.stride = 1

        self.num_slices = int(np.ceil(self.in_filters/SLICE_SHAPE[0])*np.ceil(self.out_filters/SLICE_SHAPE[1]))
        self.code = nn.Parameter(torch.randn([self.num_slices]+[CODE_SIZE]))

        # self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
        #                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.dilation = dilation
        self.padding = padding

        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        # x = self.atrous_conv(x)

        self.kernel = self.CSG(self.code)
        self.kernel = self.kernel.view(int(np.ceil(self.out_filters/SLICE_SHAPE[0])*SLICE_SHAPE[0]),int(np.ceil(self.in_filters/SLICE_SHAPE[1])*SLICE_SHAPE[1]),3,3)
        # self.kernel2 = self.kernel2[:self.out_filters2, :self.in_filters2, :self.filter_size2, :self.filter_size2]
        # self.kernel2_defined = True



        ###########################################
        ## Convolution with our kernel
        x = F.conv2d(x,self.kernel,stride=self.stride,
                               dilation=self.dilation, padding=self.dilation)


        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        for name, p in self.named_parameters():
            #print(">>>>>>>>>>..",name)
            if 'code' in name:
                p.data.normal_(1/1000)

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm, CSG):
        super(ASPP, self).__init__()

        self.CSG = CSG

        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModuleOrg(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm,CSG=CSG)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm,CSG=CSG)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm,CSG=CSG)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm, CSG):
    return ASPP(backbone, output_stride, BatchNorm, CSG)