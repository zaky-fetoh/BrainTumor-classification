import torch.utils.data as data
from data_org import *

import h5py
import numpy as np


ENCODE = {1: "meningioma", 2: "glioma", 3: "pituitary tumor"}

def readMat(path):
    """
    converts a matfile to dict of np.arrays.
    :param path: to matfile
    :return: {label: np.array, image: np.array ... }
    """
    f = h5py.File(path)
    dic = dict()
    for key, value in f["cjdata"].items():
        dic[key] = np.array(value)
    return dic


class figshare(data.Dataset):
    # Raw Dataset
    def __init__(self, onlyThisSet):
        self.onlyThisSet = onlyThisSet
    def __len__(self):
        if self.onlyThisSet is not None :
            return len(self.onlyThisSet)
        else:
            raise "Error"
    def __getitem__(self, i):
        ind = self.onlyThisSet[i]
        return readMat("dataset/"+str(ind)+".mat")

shift = lambda x : x -x.min()
scale = lambda x : x /x.max()
normalize = lambda x : scale(shift(x))
class DataFold(data.Dataset):
    def __init__(self, aug, foldInds):
        self.data = figshare(foldInds)
        self.aug = aug

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        obj = self.data.__getitem__(item)
        im = normalize(np.asarray(obj["image"],np.float))*255
        im = np.asarray(im, np.uint8)
        return self.aug(im), int(obj["label"][0,0])-1

def getfoldsArray(foldIndMat="cvind.mat"):
    foldinds = h5py.File(foldIndMat)["cvind"]
    return np.array(foldinds)

def getfoldDataSets(aug):
    fold_array = getfoldsArray()
    fold_array.shape = (fold_array.size,)
    print(fold_array.shape)
    data_list = []
    for i in range(5):
        indices = np.where(fold_array == (i+1))[0]
        data_list.append(DataFold(aug, indices))
    return data_list

def create_loader(dts, bs, ):

    tdlr = data.DataLoader(dts, bs, shuffle=True,
                           pin_memory=True, num_workers=2, )
    return tdlr


def getloaders(aug, bs=128):
    dts = getfoldDataSets(aug)
    return [create_loader(x, bs) for x in dts]


if __name__ == '__main__':
    lis = getloaders(lambda x:x)
