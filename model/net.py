import torch.nn as nn
import torch as t
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
        self.conv3 = nn.Conv2d(D*C, planes , kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes )
        self.relu = nn.LeakyReLU()

        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        if(list(out.shape) == list(x.shape)):
            out += x
        out = self.relu(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,outplanes,
                              kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes,outplanes,
                              kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu = nn.LeakyReLU()
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        if(list(out.shape) == list(x.shape)):
            out += x
        return out


class Network(nn.Module):
    def __init__(self, labelNum= 3):
        super(Network, self).__init__()
        self.convbase  = nn.Sequential(
            ResBlock(1,32),
            ResBlock(32,32),
            ResBlock(32,32),
            nn.MaxPool2d(2),
            ResBlock(32,64),
            ResBlock(64,64),
            nn.MaxPool2d(2),
            ResNextBlock(64, 128),
            ResNextBlock(128, 128),
            ResNextBlock(128, 128),
            nn.MaxPool2d(2),
            ResNextBlock(128,256),
            ResNextBlock(256,256),
        )
        self.transbase= nn.Sequential(
            nn.TransformerEncoderLayer(d_model=784,nhead=8),
            # nn.TransformerEncoderLayer(d_model=784,nhead=8),
            nn.TransformerEncoderLayer(d_model=784,nhead=8),
        )
        self.outlayer = nn.Linear(784, labelNum)

    def forward(self, x):
        x = self.convbase(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.transbase(x)
        x = x.mean(1)
        return self.outlayer(x)

countParam = lambda x: sum([p.numel() for p in x.parameters()])

if __name__ == "__main__":
    network = Network()
    inp = t.Tensor(1,1,224,224)
    out = network(inp)
    print(countParam(network))

