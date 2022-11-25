import torchvision.transforms as transf
import torch.nn.functional as f
from data_org import *
import torch.nn as nn
from model import *
import torch as t
import pickle


def save_train_hist(hist, ep_num,
                    name='hist',
                    outPath='./weights/'):
    file_name = outPath + name + str(ep_num) + '.p'
    with open(file_name, 'wb') as file:
        pickle.dump(hist, file)


def load_train_hist(ep_num, name='hist',
                    outPath='./weights/'):
    file_name = outPath + name + str(ep_num) + '.p'
    with open(file_name, 'rb')as file:
        hist = pickle.load(file)
    return hist



def save_model(net, ep_num,
               name='weight',
               outPath='./weights/'):
    file_name = outPath + name + str(ep_num) + '.pth'
    t.save(net.state_dict(), file_name)
    print('Model Saved', file_name)


def load_model(file_path=None, model=Network(),
               outPath='./weights/', name='weight', ep_num=None,
               ):
    file_path = outPath + name + str(
        ep_num) + '.pth' if not file_path else file_path
    state_dict = t.load(file_path)
    model.load_state_dict(state_dict)
    print('Model loaded', file_path)