# import extraction
import numpy as np
import data_org as dorg
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

data = dorg.getloaders(preprocess,1)

