# import extraction

import torch as t
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import data_org as dorg
import model as mdl


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

data = dorg.getloaders(preprocess, 32)

opt = optim.Adam
crit = nn.CrossEntropyLoss()


if __name__ == "__main__":
    prof = dict()
    mdl.kfoldTraining(loaders=data, profile=prof,
                      criterion=crit, optClass=opt,
                      ustep=128)
