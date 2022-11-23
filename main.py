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

for dt in data:
    for img, label in dt:
        img = img.detach().numpy()* 255
        img.shape = (224,224)
        print(img.shape, type(img), img.min(), img.max())
        plt.imshow(np.array(img, dtype= np.int))
        plt.show()
        plt.pause(1)
