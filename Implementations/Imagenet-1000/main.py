# Date: Feb 2020
# Authors: Hamed Omidvar & Vahideh Akhlaghi
# Affiliations: University of California San Diego
# Contact Information: homidvar@ucsd.edu, vakhlagh@ucsd.edu

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import torchvision.datasets as datasets
import numpy as np
import ImageNet

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import argparse
import datetime

from models import resnet as rs
#from utils import progress_bar

BATCH_SIZE = 256

#torch.manual_seed(3.2245532)

#import sys
#args = sys.argv
#data_dir = args[1]

parser = argparse.ArgumentParser(description='PyTorch IMAGENET-1000 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--iter', default=20, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--reg_mode', '-rm', action='store_true', help='regular mode')
parser.add_argument('--clg_mode', '-cl', action='store_true', help='regular mode')
parser.add_argument('--data_dir', default='/path-to-imagenet-datasets/', type=str, help='learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
# print('==> Preparing data..')
#transform_train = transforms.Compose([
#    transforms.CenterCrop(224),
#    transforms.RandomHorizontalFlip(),
#    transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#])


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transform_train =  transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])






#transform_test = transforms.Compose([
#    transforms.CenterCrop(224),
#    transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#])


transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])





data_dir = args.data_dir
result_folder = '/path-to-results-folder/results_resnet101_org_' + str(datetime.datetime.today()).replace(':','_').replace(' ','_').replace('.','_')

print("NETWORK>>>>>>>>>>>>>>>>",result_folder)
try:
    os.mkdir(result_folder)
except FileExistsError:
    print(FileExistsError)

# data_transforms = { 'train': transforms.Compose([transforms.ToTensor()]),
#                     'val'  : transforms.Compose([transforms.ToTensor(),]) }

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
#
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/',split='train', download=False)

trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform_train)
testset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform_test)

# trainset = torchvision.datasets.ImageNet('/ho-imagenet/ImageNet/',split='train', download=False)
# testset = torchvision.datasets.ImageNet('/ho-imagenet/ImageNet/',split='val', download=False)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=32)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=32)

#print("Trainset:",trainset)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

reg_mode = False

if args.reg_mode:
    reg_mode = True

print("====> REG MODE:", str(reg_mode))

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
net = rs.resnet101(org=True)
# net = GoogLeNet()
# net = densenet_imagenet(reg_mode=reg_mode)
# net = resnet56()
# net = resnet50(org = False)
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(result_folder), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(result_folder,'ckpt.t7'))
    net.load_state_dict(checkpoint['net'],strict=False)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("BEST ACC:",best_acc)


if args.clg_mode:
    print('==> Loading CLG from checkpoint..')
    assert os.path.isdir(result_folder), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(result_folder,'ckpt.t7'))
    # net.load_state_dict(checkpoint['net'],strict=False)
    #print(checkpoint)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if 'CLG' in k}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    net.load_state_dict(model_dict)

#     net.load_state_dict(checkpoint['net'],strict=False)

    for name, param in net.named_parameters():
        if 'CLG' in name:
            print("NO GRAD FOR:", name, '==> Reducing Params by:', np.prod(param.data.shape))
            param.requires_grad = False



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print("NUMBER OF TRAINABLE PARAMETERS:",params)


# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])


for name, param in net.named_parameters():
    if param.requires_grad:
        print (name,
                param.data.shape,'==>>', np.prod(param.data.shape)
                )

#if not args.resume:
    #folder = './results/'
    #for the_file in os.listdir(folder):
        #file_path = os.path.join(folder, the_file)
        # try:
        #     if os.path.isfile(file_path):
        #         os.unlink(file_path)
        #     #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        # except Exception as e:
        #     print(e)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if train_loss == 0:
            print(inputs.size())
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # try:
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 10 == 0:
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # except:
        #     print("Error :(")
    return 100.*correct/total, train_loss/BATCH_SIZE

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 10 == 0:
                print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint.
    acc = 100.*correct/total
    #if acc > best_acc:
    #if True:
    print('Saving..')
    state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
    }
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    torch.save(state, os.path.join(result_folder,'ckpt.t7'))
    best_acc = acc


    return 100.*correct/total, test_loss/BATCH_SIZE

steps = [30, 30, 30, 10]
# steps = [21,60,10]
epoch = start_epoch
# K = int(args.iter)
for i in range(len(steps)):
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0.00001 and i != 0:
            param_group['lr'] =  param_group['lr']/10
        print("Learning Rate:", param_group['lr'])
    #K = np.ceil(10//2**i)
#    K = int(max(int(args.iter)//(2)**i,50))
    K = steps[i]
    for _ in range(K):
        train_acc, train_loss = train(epoch)
        test_acc, test_loss = test(epoch)
        with open(result_folder+'/train_acc.txt','a+') as f:
            f.write(str(train_acc)+'; ')
        with open(result_folder+'/train_loss.txt','a+') as f:
            f.write(str(train_loss)+'; ')
        with open(result_folder+'/test_acc.txt','a+') as f:
            f.write(str(test_acc)+'; ')
        with open(result_folder+'/test_loss.txt','a+') as f:
            f.write(str(test_loss)+'; ')
        epoch += 1


    # print('==> Resuming from checkpoint..')
    # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('./checkpoint/ckpt.t7')
    # net.load_state_dict(checkpoint['net'],strict=False)
    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']
    # print("BEST ACC:",best_acc)
