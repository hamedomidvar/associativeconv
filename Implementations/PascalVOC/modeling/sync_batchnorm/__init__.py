# Date: May 2019
# Authors: Omitted for anonymity
# Affiliations: Omitted for anonymity
# Contact Information: Omitted for anonymity
# Original Repository: https://github.com/jfzhang95/pytorch-deeplab-xception


### Original Repo:
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

from .batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
from .replicate import DataParallelWithCallback, patch_replication_callback
