import math
import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import math
import numpy as np
import torch.nn.functional as F

#########
CODE_SIZE = 128
SLICE_SHAPE = [16,16,3,3]
#########

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, CSG, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()

        self.stride = stride
        self.dilation = dilation


        self.CSG = CSG

        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm(planes)
        #self.conv2 = conv3x3(width, width, CSG, stride, groups, dilation)
        self.stride2 = stride
        self.filter_size2 = 3
        self.in_filters2 = planes
        self.out_filters2 = planes
        self.num_slices2 = int(np.ceil(self.in_filters2/SLICE_SHAPE[0])*np.ceil(self.out_filters2/SLICE_SHAPE[1]))
        self.code2 = nn.Parameter(torch.randn([self.num_slices2]+[CODE_SIZE]))
        self.kernel2 = None
        self.kernel2_defined = False


        self.bn2 = BatchNorm(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #out = self.conv2(out)
        #######################
        self.kernel2 = self.CSG(self.code2)
        self.kernel2 = self.kernel2.view(int(np.ceil(self.out_filters2/SLICE_SHAPE[0])*SLICE_SHAPE[0]),int(np.ceil(self.in_filters2/SLICE_SHAPE[1])*SLICE_SHAPE[1]),3,3)
        self.kernel2 = self.kernel2[:self.out_filters2, :self.in_filters2, :self.filter_size2, :self.filter_size2]
        self.kernel2_defined = True



        ###########################################
        ## Convolution with our kernel
        out = F.conv2d(out,self.kernel2,stride=self.stride,
                               dilation=self.dilation, padding=self.dilation)


        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BottleneckOrg(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True, org=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError


        if not org:
            #############################################
            ## Here is where the CSG is defined.
            self.CSG = nn.Linear(CODE_SIZE,np.prod(SLICE_SHAPE),bias=False)
        else:
            self.CSG = None

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.CSG, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.CSG, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.CSG, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, self.CSG, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        print('==> Loading from checkpoint..')

        # for dirname, dirnames, filenames in os.walk('.'):
        #     # print path to all subdirectories first.
        #     for subdirname in dirnames:
        #         print(os.path.join(dirname, subdirname))
        # chk_dir = './modeling/backbone/resnetcsg_checkpoint'
        chk_dir = '/opt/repo/deeplab_csg2/modeling/backbone/resnetcsg_checkpoint'

        assert os.path.isdir(chk_dir), 'Error: no checkpoint directory found!'
        # checkpoint = torch.load('opt/repo/deeplabcsg/modeling/backbone/resnetcsg_checkpoint/ckpt.t7')
        # self.load_state_dict(checkpoint['net'],strict=False)


        # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        pretrain_dict = torch.load(chk_dir+'/ckpt.t7')
        model_dict = {}
        state_dict = self.state_dict()
        # for k in state_dict:
        #     print ("STATE DICT k:", k)
        for k, v in pretrain_dict['net'].items():
            # print ("k:",k[7:])
            if k[7:] in state_dict:
                print(k[7:]," LOADED")
                model_dict[k[7:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)



def ResNet101_csg(output_stride, BatchNorm, pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model

def ResNet50_csg(output_stride, BatchNorm, pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model

def ResNet18_csg(output_stride, BatchNorm, pretrained=True):
    model = ResNet(Bottleneck, [2, 2, 2, 2], output_stride, BatchNorm, pretrained=pretrained)
    return model

if __name__ == "__main__":
    import torch
    model = ResNet101_csg(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
