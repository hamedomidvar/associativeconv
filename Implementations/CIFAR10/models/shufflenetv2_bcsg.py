'''ShuffleNetV2 in PyTorch.

See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
######
CODE_SIZE = 16
SLICE_SHAPE = [16,16,1,1]
#########


class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5):
        super(BasicBlock, self).__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = F.relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out

class BasicBlockCSG(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5, CSG=None):
        super(BasicBlockCSG, self).__init__()

        self.CSG = CSG

        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)

#         self.conv1 = nn.Conv2d(in_channels, in_channels,
#                                kernel_size=1, bias=False)

        self.filter_size1 = 1
        self.in_filters1 = in_channels
        self.out_filters1 =in_channels
        self.num_slices1 = int(np.ceil(self.in_filters1/SLICE_SHAPE[0])*np.ceil(self.out_filters1/SLICE_SHAPE[1]))
        self.code1 = torch.nn.Parameter(torch.randn([self.num_slices1]+[CODE_SIZE]))
        self.kernel1 = None
        self.kernel1_defined = False

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)

        #########################################
        ## Updating the kernel
        self.kernel1 = self.CSG(self.code1)
        self.kernel1 = self.kernel1.view(int(np.ceil(self.out_filters1/SLICE_SHAPE[0])*SLICE_SHAPE[0]), int(np.ceil(self.in_filters1/SLICE_SHAPE[1])*SLICE_SHAPE[1]), 1,1)
        self.kernel1 = self.kernel1[:self.out_filters1, :self.in_filters1, :self.filter_size1, :self.filter_size1]
        self.kernel1_defined = True

        #         out = F.relu(self.bn1(self.conv1(x2)))

        out = F.relu(self.bn1(F.conv2d(x2,self.kernel1,padding=0)))

        out = self.bn2(self.conv2(out))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        # left
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # right
        self.conv3 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_channels)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channels)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # left
        out1 = self.bn1(self.conv1(x))
        out1 = F.relu(self.bn2(self.conv2(out1)))
        # right
        out2 = F.relu(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        out2 = F.relu(self.bn5(self.conv5(out2)))
        # concat
        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out

class DownBlockCSG(nn.Module):
    def __init__(self, in_channels, out_channels, CSG):
        super(DownBlockCSG, self).__init__()
        mid_channels = out_channels // 2
        # left

        self.CSG = CSG



        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)


#         self.conv2 = nn.Conv2d(in_channels, mid_channels,
#                                kernel_size=1, bias=False)

        self.filter_size2 = 1
        self.in_filters2 = in_channels
        self.out_filters2 =mid_channels
        self.num_slices2 = int(np.ceil(self.in_filters2/SLICE_SHAPE[0])*np.ceil(self.out_filters2/SLICE_SHAPE[1]))
        self.code2 = torch.nn.Parameter(torch.randn([self.num_slices2]+[CODE_SIZE]))
        self.kernel2 = None
        self.kernel2_defined = False

        self.bn2 = nn.BatchNorm2d(mid_channels)
        # right
#         self.conv3 = nn.Conv2d(in_channels, mid_channels,
#                                kernel_size=1, bias=False)


        self.filter_size3 = 1
        self.in_filters3 = in_channels
        self.out_filters3 =mid_channels
        self.num_slices3 = int(np.ceil(self.in_filters3/SLICE_SHAPE[0])*np.ceil(self.out_filters3/SLICE_SHAPE[1]))
        self.code3 = torch.nn.Parameter(torch.randn([self.num_slices3]+[CODE_SIZE]))
        self.kernel3 = None
        self.kernel3_defined = False



        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_channels)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channels)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # left
        out1 = self.bn1(self.conv1(x))

        #########################################
        ## Updating the kernel
        self.kernel2 = self.CSG(self.code2)
        self.kernel2 = self.kernel2.view(int(np.ceil(self.out_filters2/SLICE_SHAPE[0])*SLICE_SHAPE[0]), int(np.ceil(self.in_filters2/SLICE_SHAPE[1])*SLICE_SHAPE[1]), 1,1)
        self.kernel2 = self.kernel2[:self.out_filters2, :self.in_filters2, :self.filter_size2, :self.filter_size2]
        self.kernel2_defined = True


#         out1 = F.relu(self.bn2(self.conv2(out1)))
        out1 = F.relu(self.bn2(F.conv2d(out1,self.kernel2,padding=0)))


        # right
#         out2 = F.relu(self.bn3(self.conv3(x)))
        #########################################
        ## Updating the kernel
        self.kernel3 = self.CSG(self.code3)
        self.kernel3 = self.kernel3.view(int(np.ceil(self.out_filters3/SLICE_SHAPE[0])*SLICE_SHAPE[0]), int(np.ceil(self.in_filters3/SLICE_SHAPE[1])*SLICE_SHAPE[1]), 1,1)
        self.kernel3 = self.kernel3[:self.out_filters3, :self.in_filters3, :self.filter_size3, :self.filter_size3]
        self.kernel3_defined = True

        out2 = F.relu(self.bn3(F.conv2d(x,self.kernel3,padding=0)))



        out2 = self.bn4(self.conv4(out2))
        out2 = F.relu(self.bn5(self.conv5(out2)))
        # concat
        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out



class ShuffleNetV2(nn.Module):
    def __init__(self, net_size, org =False):
        super(ShuffleNetV2, self).__init__()
        out_channels = configs[net_size]['out_channels']
        num_blocks = configs[net_size]['num_blocks']


        self.org = org
        if not org:
            #############################################
            ## Here is where the CSG is defined.
            self.CSG = torch.nn.Linear(CODE_SIZE,np.prod(SLICE_SHAPE), bias=False)

            #### BINARY CSG:
            nn.init.kaiming_normal_(self.CSG.weight)
            self.CSG.weight.data = torch.sign(self.CSG.weight.data)*0.5
            self.CSG.weight.requires_grad_(False)
        else:
            self.CSG = None



        self.conv1 = nn.Conv2d(3, 24, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])


#         self.conv2 = nn.Conv2d(out_channels[2], out_channels[3],
#                                kernel_size=1, stride=1, padding=0, bias=False)

        self.filter_size2 = 1
        self.in_filters2 = out_channels[2]
        self.out_filters2 =out_channels[3]
        self.num_slices2 = int(np.ceil(self.in_filters2/SLICE_SHAPE[0])*np.ceil(self.out_filters2/SLICE_SHAPE[1]))
        self.code2 = torch.nn.Parameter(torch.randn([self.num_slices2]+[CODE_SIZE]))
        self.kernel2 = None
        self.kernel2_defined = False

        self.bn2 = nn.BatchNorm2d(out_channels[3])
        self.linear = nn.Linear(out_channels[3], 10)

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.max_pool2d(out, 3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        #########################################
        ## Updating the kernel
        self.kernel2 = self.CSG(self.code2)
        self.kernel2 = self.kernel2.view(int(np.ceil(self.out_filters2/SLICE_SHAPE[0])*SLICE_SHAPE[0]), int(np.ceil(self.in_filters2/SLICE_SHAPE[1])*SLICE_SHAPE[1]), 1,1)
        self.kernel2 = self.kernel2[:self.out_filters2, :self.in_filters2, :self.filter_size2, :self.filter_size2]
        self.kernel2_defined = True

#         out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn2(F.conv2d(out,self.kernel2,padding=0)))


        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


configs = {
    0.5: {
        'out_channels': (48, 96, 192, 1024),
        'num_blocks': (3, 7, 3)
    },

    1: {
        'out_channels': (116, 232, 464, 1024),
        'num_blocks': (3, 7, 3)
    },
    1.5: {
        'out_channels': (176, 352, 704, 1024),
        'num_blocks': (3, 7, 3)
    },
    2: {
        'out_channels': (224, 488, 976, 2048),
        'num_blocks': (3, 7, 3)
    }
}


def test():
    net = ShuffleNetV2(net_size=0.5)
    x = torch.randn(3, 3, 32, 32)
    y = net(x)
    print(y.shape)


# test()
