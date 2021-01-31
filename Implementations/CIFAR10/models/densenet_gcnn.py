import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from torch.autograd import Variable
from groupy.gconv.pytorch_gconv import p4_conv, p4m_conv

#############################
CODE_SIZE = 72
SLICE_SHAPE = [12,12,3,3]
#############################

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, CSG, reg_mode=False):
        super(Bottleneck, self).__init__()
        self.reg_mode = reg_mode

        self.CSG = CSG

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(4*growth_rate)

        ###############################################################
        #### The following covolutional layer is replaced by our CSG generated kernel
        #self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

        ###############################################################
        ## Here we set things up for the CSG including defining a code matrix for each layer
        self.filter_size2 = 3
        self.in_filters2 = 4*growth_rate
        self.out_filters2 = growth_rate
        self.num_slices2 = int(np.ceil(self.in_filters2/SLICE_SHAPE[0])*np.ceil(self.out_filters2/SLICE_SHAPE[1]))
        self.code2 = torch.nn.Parameter(torch.randn([self.num_slices2]+[CODE_SIZE]))
        self.kernel2 = None
        self.kernel2_defined = False

    def forward(self, x):

        out = self.conv1(F.relu(self.bn1(x)))

        #########################################
        ## Updating the kernel
        self.kernel2 = self.CSG(self.code2)
        self.kernel2 = self.kernel2.view(int(np.ceil(self.out_filters2/SLICE_SHAPE[0])*SLICE_SHAPE[0]),int(np.ceil(self.in_filters2/SLICE_SHAPE[1])*SLICE_SHAPE[1]),3,3)
        self.kernel2 = self.kernel2[:self.out_filters2, :self.in_filters2, :self.filter_size2, :self.filter_size2]
        self.kernel2_defined = True


        ###########################################
        ### This is replaced by our kernel
        #out = self.conv2(F.relu(self.bn2(out)))


        ###########################################
        ## Convolution with our kernel
        out = F.conv2d(F.relu(self.bn2(out)),self.kernel2,padding=1)
        out = torch.cat([out,x], 1)
        return out


class BottleneckOrg(nn.Module):
    def __init__(self, in_planes, growth_rate, CSG, reg_mode=False):
        super(BottleneckOrg, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)


    def forward(self, x):

        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class BottleneckG(nn.Module):
    def __init__(self, in_planes, growth_rate, CSG, reg_mode=False):
        super(BottleneckG, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = p4_conv.P4ConvP4(in_channels=in_planes//4, out_channels=4*growth_rate//4, ksize=1)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
#         self.conv1 = p4_conv.P4ConvP4(in_planes//4, 4*growth_rate//4, ksize=1)
        self.out_ch1 = 4*growth_rate

        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        # print("4times growth rate:",4*growth_rate)
        self.conv2 = p4_conv.P4ConvP4(in_channels=4*growth_rate//4, out_channels=growth_rate//4, ksize=3, pad=1)
        self.out_ch2 = growth_rate


    def forward(self, x):

        out = self.conv1(F.relu(self.bn1(x)))
#         print(out.size())
        out = out.view(-1,self.out_ch1,out.size()[-2],out.size()[-1])
        out = self.conv2(F.relu(self.bn2(out)))
        # print(out.size(),self.out_ch2)
        out = out.view(-1,self.out_ch2,out.size()[-2],out.size()[-1])
        out = torch.cat([out,x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, reg_mode, last = False):
        super(Transition, self).__init__()

        self.last = last

        self.reg_mode = reg_mode

        self.bn = nn.BatchNorm2d(in_planes)
        if not last:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)


    def forward(self, x):
        out = F.relu(self.bn(x))
        if not self.last:
            out = self.conv(out)
            #out = F.conv2d(out,self.kernel1,padding=1)
            out = F.avg_pool2d(out, 2)
        else:
            out = F.avg_pool2d(out, 8)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, reg_mode = False, org = False):
        super(DenseNet, self).__init__()

        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        #################################################
        if not org:
            #############################################
            ## Here is where the CSG is defined.
            self.CSG = torch.nn.Linear(CODE_SIZE,np.prod(SLICE_SHAPE),bias=False)
        else:
            self.CSG = None


        #################################################
        ## The following is based on densenet

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks, reg_mode=reg_mode)
        num_planes += nblocks*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes, reg_mode=reg_mode)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks, reg_mode=reg_mode)
        num_planes += nblocks*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes, reg_mode=reg_mode)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks, reg_mode=reg_mode)
        num_planes += nblocks*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes, reg_mode=reg_mode, last = True)

        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock, reg_mode=False):
        layers = []
        for i in range(nblock):
            ###################################################
            ## We merely pass the reference of CSG to all blocks
            layers.append(block(in_planes, self.growth_rate, self.CSG, reg_mode=reg_mode))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def densenet_cifar_g(reg_mode=False):
    # L = 40 --> (40-4)/3 = 12 --> Using Bottleneck --> 12/2 = 6 = nblock
    return DenseNet(BottleneckG, 6, growth_rate=48, reg_mode=reg_mode)

def densenet_cifar_org(reg_mode=False):
    return DenseNet(BottleneckOrg, 6, growth_rate=48, reg_mode=reg_mode, org = True)
