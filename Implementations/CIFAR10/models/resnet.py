import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .res_utils import DownsampleA, DownsampleC, DownsampleD
import math
import numpy as np

###############################
CODE_SIZE = 72
SLICE_SHAPE = [12,12,3,3]
###############################

class ResNetBasicblock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, CSG, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()
        self.stride = stride
        self.reg_mode = False
        self.CSG = CSG
        #self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.filter_size = 3
        self.in_filters = inplanes
        self.out_filters = planes
        self.num_slices = int(np.ceil(self.in_filters/SLICE_SHAPE[0])*np.ceil(self.out_filters/SLICE_SHAPE[1]))
        self.code = torch.nn.Parameter(torch.randn([self.num_slices]+[CODE_SIZE]))
        #self.register_parameter('code'+str(in_planes)+"_"+str(4*growth_rate),self.code)
        self.kernel1 = None
        self.kernel1_defined = False

        if self.reg_mode:
            self.kernel1 = torch.nn.Parameter(torch.randn(self.out_filters, self.in_filters, self.filter_size, self.filter_size))
            # self.kernel1_defined = True

        #self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.filter_size2 = 3
        self.in_filters2 = planes
        self.out_filters2 = planes
        self.num_slices2 = int(np.ceil(self.in_filters2/SLICE_SHAPE[0])*np.ceil(self.out_filters2/SLICE_SHAPE[1]))
        self.code2 =torch.nn.Parameter(torch.randn([self.num_slices2]+[CODE_SIZE]))
        self.kernel2 = None
        self.kernel2_defined = False


        if self.reg_mode:
            self.kernel2 = torch.nn.Parameter(torch.randn(self.out_filters2, self.in_filters2, self.filter_size2, self.filter_size2))


        self.downsample = downsample

    def forward(self, x):
        if not self.kernel1_defined  and self.reg_mode:
            k1 = self.CSG(self.code)
            k1 = k1.view(int(np.ceil(self.out_filters/SLICE_SHAPE[0])*SLICE_SHAPE[0]),int(np.ceil(self.in_filters/SLICE_SHAPE[1])*SLICE_SHAPE[1]),3,3)
            k1 = k1[:self.out_filters, :self.in_filters, :self.filter_size, :self.filter_size]
            self.kernel1 = torch.nn.Parameter(k1)
        if not self.reg_mode:
            self.kernel1 = self.CSG(self.code)
            self.kernel1 = self.kernel1.view(int(np.ceil(self.out_filters/SLICE_SHAPE[0])*SLICE_SHAPE[0]),int(np.ceil(self.in_filters/SLICE_SHAPE[1])*SLICE_SHAPE[1]),3,3)
            self.kernel1 = self.kernel1[:self.out_filters, :self.in_filters, :self.filter_size, :self.filter_size]
        self.kernel1_defined = True

        residual = x

        basicblock =  F.conv2d(x,self.kernel1,padding=1,stride=self.stride)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        if not self.kernel2_defined  and self.reg_mode:
            k2 = self.CSG(self.code2)
            k2 = k2.view(int(np.ceil(self.out_filters2/SLICE_SHAPE[0])*SLICE_SHAPE[0]),int(np.ceil(self.in_filters2/SLICE_SHAPE[1])*SLICE_SHAPE[1]),3,3)
            k2 = k2[:self.out_filters2, :self.in_filters2, :self.filter_size2, :self.filter_size2]
            self.kernel2 = torch.nn.Parameter(k2)
        if not self.reg_mode:
            self.kernel2 = self.CSG(self.code2)
            self.kernel2 = self.kernel2.view(int(np.ceil(self.out_filters2/SLICE_SHAPE[0])*SLICE_SHAPE[0]),int(np.ceil(self.in_filters2/SLICE_SHAPE[1])*SLICE_SHAPE[1]),3,3)
            self.kernel2 = self.kernel2[:self.out_filters2, :self.in_filters2, :self.filter_size2, :self.filter_size2]
        self.kernel2_defined = True

        basicblock =  F.conv2d(basicblock, self.kernel2, padding=1, stride=1)
        #basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)




###########################


class ResNetBasicblockOrg(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, CSG, stride=1, downsample=None):
        super(ResNetBasicblockOrg, self).__init__()
        self.stride = stride


        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)


        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """
    def __init__(self, block, depth, num_classes, org = False):
        """ Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
        """
        super(CifarResNet, self).__init__()

        #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

        self.num_classes = num_classes

        #################################################
        if not org:
            self.CSG = torch.nn.Linear(CODE_SIZE,np.prod(SLICE_SHAPE),bias=False)

            #### BINARY CSG:
            # nn.init.kaiming_normal_(self.CSG.weight)
            # self.CSG.weight.data = torch.sign(self.CSG.weight.data)*0.5
            # self.CSG.weight.requires_grad_(False)


        else:
            self.CSG = None


        self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, self.CSG, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, self.CSG, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, self.CSG, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

        for name, p in self.named_parameters():
            if 'code' in name:
                p.data.normal_(1/1000)

    def _make_layer(self, block, planes, blocks, CSG, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, self.CSG, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.CSG))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def resnet20(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
    num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 20, num_classes)
    return model

def resnet20_org(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
    num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblockOrg, 20, num_classes, org=True)
    return model

def resnet56(num_classes=10):
    """Constructs a ResNet-56 model for CIFAR-10 (by default)
    Args:
    num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 56, num_classes)
    return model


def resnet56_org(num_classes=10):
    """Constructs a ResNet-56 model for CIFAR-10 (by default)
    Args:
    num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblockOrg, 56, num_classes, org=True)
    return model
