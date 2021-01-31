# Date: Feb 2020
# Authors: Hamed Omidvar & Vahideh Akhlaghi
# Affiliations: University of California San Diego
# Contact Information: homidvar@ucsd.edu, vakhlagh@ucsd.edu

from __future__ import print_function

import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import argparse

import torchvision
import torchvision.transforms as transforms

## PLEASE IMPORT THE CORRECT MODEL HERE
from models.densenet import *


from utils import progress_bar

#####################################
# Please set these for each model in the model files:
# CODE_SIZE = 128 or 72
# SLICE_SHAPE = [16,16,3,3]  or [12,12,3,3]
#####################################
BATCH_SIZE =  128 # We used 128 for all DenseNet models and their CSG versions and 192 for all ResNet models and their CSG versions
#####################################


#####################################
## INPUT PARAMETERS
#####################################

parser = argparse.ArgumentParser(description='Training of CIFAR-10 using CSG augmented CNNs')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--iter', default=100, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--csg_mode', '-cl', action='store_true', help='CSG mode')
######################################
### We don't use reg_mode in this implementation so you can safely ignore it.
## It is used for the case when the network is first pretrained with CSG and then
## the training is continued with ordinary kernels but initialized with trained slices so far.
parser.add_argument('--reg_mode', '-rm', action='store_true', help='regular mode')
args = parser.parse_args()



device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data preparation
print('-----> PRE-PROCESSING THE DATA')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

#print("Trainset:",trainset)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

reg_mode = False

if args.reg_mode:
    reg_mode = True

print("-----> REG MODE:", str(reg_mode))





print('-----> MODEL INITIALIZATION')
######################################
net = densenet_cifar(reg_mode=reg_mode)
#net = densenet_cifar_org(reg_mode=reg_mode)
#net = resnet56()
#net = resnet56_org()
#net = resnet20()
#net = resnet20_org()
#...
######################################


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'],strict=False)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("BEST ACC:",best_acc)


if args.csg_mode:
    print('==> Loading CSG from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: No Checkpoint directory was found.'
    # 1. Loading the dictionary
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if 'CSG' in k}
    # 2. Overwrititing entries in the existing state dict with parameters of CSG
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    net.load_state_dict(model_dict)

    for name, param in net.named_parameters():
        if 'CSG' in name:
            print("NO GRAD FOR:", name, '==> Reduing Params by:', np.prod(param.data.shape))
            param.requires_grad = False



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


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

if not args.resume:
    folder = './results/'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


# Training
def train(epoch):
    t_0 = time.time()
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return 100.*correct/total, train_loss/BATCH_SIZE, time.time()-t_0

def test(epoch):
    t_0 = time.time()
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


    return 100.*correct/total, test_loss/BATCH_SIZE, time.time()-t_0


epoch = start_epoch
K = int(args.iter)
for i in range(3):
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0.00001 and i != 0:
            param_group['lr'] =  param_group['lr']/10
        print("Learning Rate:", param_group['lr'])
    #K = np.ceil(10//2**i)
    K = int(max(int(args.iter)//(2)**i,50))
    for _ in range(K):
        train_acc, train_loss, train_time = train(epoch)
        test_acc, test_loss, test_time = test(epoch)
        with open('./results/train_acc.txt','a+') as f:
            f.write(str(train_acc)+'; ')
        with open('./results/train_loss.txt','a+') as f:
            f.write(str(train_loss)+'; ')
        with open('./results/test_acc.txt','a+') as f:
            f.write(str(test_acc)+'; ')
        with open('./results/test_loss.txt','a+') as f:
            f.write(str(test_loss)+'; ')
        with open('./results/train_time.txt','a+') as f:
            f.write(str(train_time)+'; ')
        with open('./results/test_time.txt','a+') as f:
            f.write(str(test_time)+'; ')
        epoch += 1
