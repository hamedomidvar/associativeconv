'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

######
CODE_SIZE = 16
SLICE_SHAPE = [16,16,1,1]
#########

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out
    

class BlockCSG(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, CSG):
        super(BlockCSG, self).__init__()
        self.stride = stride

        self.CSG = CSG
        
        planes = expansion * in_planes
        
        
        
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.filter_size1 = 1
        self.in_filters1 = in_planes
        self.out_filters1 = planes
        self.num_slices1 = int(np.ceil(self.in_filters1/SLICE_SHAPE[0])*np.ceil(self.out_filters1/SLICE_SHAPE[1]))
        self.code1 = torch.nn.Parameter(torch.randn([self.num_slices1]+[CODE_SIZE]))
        self.kernel1 = None
        self.kernel1_defined = False
        
        
        self.bn1 = nn.BatchNorm2d(planes)

        
        

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        
        
        #########################################
        ## Updating the kernel
        self.kernel1 = self.CSG(self.code1)
        self.kernel1 = self.kernel1.view(int(np.ceil(self.out_filters1/SLICE_SHAPE[0])*SLICE_SHAPE[0]), int(np.ceil(self.in_filters1/SLICE_SHAPE[1])*SLICE_SHAPE[1]), 1,1)
        self.kernel1 = self.kernel1[:self.out_filters1, :self.in_filters1, :self.filter_size1, :self.filter_size1]
        self.kernel1_defined = True
        
#         out = F.relu(self.bn1(self.conv1(x)))
        
        out = F.relu(self.bn1(F.conv2d(x,self.kernel1,padding=0)))
        
        out = F.relu(self.bn2(self.conv2(out)))

#         out = F.relu(self.bn2(F.conv2d(out,self.kernel2,padding=1)))
        
        
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, org = False): # NOTE : 1000 > 10
        super(MobileNetV2, self).__init__()
        
        self.org = org
        if not org:
            #############################################
            ## Here is where the CSG is defined.
            self.CSG = torch.nn.Linear(CODE_SIZE,np.prod(SLICE_SHAPE),bias=False)
        else:
            self.CSG = None
        
        
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layers = self._make_layers(in_planes=32)
        
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                if self.org:
                    layers.append(Block(in_planes, out_planes, expansion, stride))
                else:
                    layers.append(BlockCSG(in_planes, out_planes, expansion, stride, CSG = self.CSG))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
