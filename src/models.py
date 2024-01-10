# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 18/08/2022 14:19
# @Author       : tl22089
# @File         : models.py
# @Affiliation  : University of Bristol
import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.input = nn.Linear(args.ws * 3, 512)
        if args.type != 'metric':
            self.fc = nn.Linear(in_features=512, out_features=args.bs_classes)
        else:
            self.fc = nn.Linear(in_features=512, out_features=args.out_dim)

    def forward(self, x):
        x = F.relu(self.input(x))
        return self.fc(x)


# class ResNet(nn.Module):
#     def __init__(self, args):
#         super(ResNet, self).__init__()
#         self.args = args
#         self.resnet = models.resnet18(pretrained=False)
#         self.resnet_wo_fc = nn.Sequential(*(list(self.resnet.children())[:-1]))
#         if args.type != 'metric':
#             self.fc = nn.Linear(in_features=512, out_features=args.bs_classes)
#         else:
#             self.fc = nn.Linear(in_features=512, out_features=args.out_dim)
#
#     def _convert_data_size(self, x):
#         xr = x[:, :self.args.ws].view(-1, 1, 32, 32)
#         xi = x[:, self.args.ws:2 * self.args.ws].view(-1, 1, 32, 32)
#         xn = x[:, self.args.ws * 2:].view(-1, 1, 32, 32)
#         return torch.concat([xr, xi, xn], dim=1)
#
#     def forward(self, x):
#         x = self._convert_data_size(x)
#         x = self.resnet_wo_fc(x)
#         x = torch.flatten(x, 1)
#         return self.fc(x)


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 64, 3)
        self.fc1 = nn.Linear(2304, 512)
        if args.type != 'metric':
            self.fc = nn.Linear(in_features=512, out_features=args.bs_classes)
        else:
            self.fc = nn.Linear(in_features=512, out_features=args.out_dim)

    def _convert_data_size(self, x):
        xr = x[:, :self.args.ws].view(-1, 1, 32, 32)
        xi = x[:, self.args.ws:2 * self.args.ws].view(-1, 1, 32, 32)
        xn = x[:, self.args.ws * 2:].view(-1, 1, 32, 32)
        return torch.concat([xr, xi, xn], dim=1)

    def forward(self, x):
        x = self._convert_data_size(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc(x)


class CNNFashion(nn.Module):
    def __init__(self, args):
        super(CNNFashion, self).__init__()
        self.args = args
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        if args.type != 'metric':
            self.out = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features=7 * 7 * 32, out_features=args.bs_classes)
            )
        else:
            self.out = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features=7 * 7 * 32, out_features=args.out_dim)
            )

    def _convert_data_size(self, x):
        xr = x[:, :self.args.ws].view(-1, 1, 32, 32)
        xi = x[:, self.args.ws:2 * self.args.ws].view(-1, 1, 32, 32)
        xn = x[:, self.args.ws * 2:].view(-1, 1, 32, 32)
        return torch.concat([xr, xi, xn], dim=1)

    def forward(self, x):
        x = self._convert_data_size(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.out(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes + i * growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DenseNet(nn.Module):
    def __init__(self, args, depth, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet, self).__init__()
        self.args = args
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n / 2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.in_planes = in_planes

        if args.type != 'metric':
            self.fc = nn.Linear(in_features=in_planes, out_features=args.bs_classes)
        else:
            self.fc = nn.Linear(in_features=in_planes, out_features=args.out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _convert_data_size(self, x):
        xr = x[:, :self.args.ws].view(-1, 1, 32, 32)
        xi = x[:, self.args.ws:2 * self.args.ws].view(-1, 1, 32, 32)
        xn = x[:, self.args.ws * 2:].view(-1, 1, 32, 32)
        return torch.concat([xr, xi, xn], dim=1)

    def forward(self, x):
        out = self._convert_data_size(x)
        out = self.conv1(out)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)
