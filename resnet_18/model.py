import onnx

import torch
import torch.nn as nn
from torch.nn import functional as F

import brevitas.nn as qnn

from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import brevitas.onnx as bo

import sys
import numpy as np
import time
import utils
import glob
import random
import logging
#import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.autograd import Variable



return_quant = True
bit_width_input = 8
bit_width_weight = 8
bit_width_weight_b = 2
bit_width_act = 2
bit_width_pool = 8

class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = qnn.QuantConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, weight_bit_width=bit_width_weight_b, return_quant_tensor=return_quant)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = qnn.QuantReLU(bit_width=bit_width_act, return_quant_tensor=return_quant)
        self.conv2 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, weight_bit_width=bit_width_weight_b, return_quant_tensor=return_quant)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = qnn.QuantReLU(bit_width=bit_width_act, return_quant_tensor=return_quant)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        
        return self.relu2(self.relu2(x) + self.relu2(output))


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = qnn.QuantConv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1, bias=False, weight_bit_width=bit_width_weight_b, return_quant_tensor=return_quant)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = qnn.QuantReLU(bit_width=bit_width_act, return_quant_tensor=return_quant)
        self.conv2 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1, bias=False, weight_bit_width=bit_width_weight_b, return_quant_tensor=return_quant)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = qnn.QuantReLU(bit_width=bit_width_act, return_quant_tensor=return_quant)
        self.extra = nn.Sequential(
            qnn.QuantConv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0, bias=False, weight_bit_width=bit_width_weight, return_quant_tensor=return_quant),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        extra_x = self.relu1(extra_x)
        output = self.conv1(x)
        out = self.bn1(output)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu1(out)
        return self.relu2(extra_x + out)


class ResNet18(nn.Module):
    def __init__(self, criterion):
        super(ResNet18, self).__init__()
        self._criterion = criterion
        #self.quant_inp = qnn.QuantIdentity(bit_width=bit_width_input, return_quant_tensor=return_quant)
        self.conv1 = qnn.QuantConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, weight_bit_width=bit_width_weight, return_quant_tensor=return_quant)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = qnn.QuantReLU(bit_width=bit_width_act, return_quant_tensor=return_quant)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1), RestNetBasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]), RestNetBasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]), RestNetBasicBlock(256, 256, 1))
        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]), RestNetBasicBlock(512, 512, 1))
        #self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.avgpool = qnn.QuantAvgPool2d(kernel_size=7, stride=1, bit_width=bit_width_pool)
        self.fc = qnn.QuantLinear(512, 1000, bias=False, weight_bit_width=bit_width_weight)

    def forward(self, x):
        #out = self.quant_inp(x)
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        #out = out.view(out.size(0), -1)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
    
    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)
