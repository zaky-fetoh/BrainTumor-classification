import torch as t
import torch.nn as nn
import math

class ResNextBlock(nn.Module):
    #Resnextblock
    def __init__(self, inplanes, planes, baseWidth=1,
                 cardinality=32, stride=1, downsample=None):
        """ Constructor
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width defult To One.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(ResNextBlock, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C)
        self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D*C)
        self.conv3 = nn.Conv2d(D*C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.LeakyReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,outplanes,
                              kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(inplanes,outplanes,
                              kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out + x


