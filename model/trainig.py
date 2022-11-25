import torchvision.transforms as transf
import torch.nn.functional as f
from data_org import *
import torch.nn as nn
from model import *
import torch as t
import pickle

import torchmetrics.classification as clfmet


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

def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with t.no_grad():
        output = output.argmax(1)
        matchs = output == target
        return matchs.sum().float() / matchs.size(0)





def train_(net, train_loader, criterion, opt_fn, ustep,
           device=t.device('cuda' if t.cuda.is_available() else 'cpu'),
           ):
    acc = clfmet.Accuracy(top_k=1)
    llis, alis, samples = list(), list(), 0
    for imgs, target in train_loader:
        #Move to Device
        imgs = imgs.to(device=device)
        target = target.to(device=device)

        samples += imgs.shape[0]
        pred = net(imgs)

        loss = criterion(pred, target)
        loss.backward()
        if samples >= ustep:
            print('netupdated:', samples)
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            opt_fn.step()
            opt_fn.zero_grad()
            samples = 0

        llis.append(loss.item())
        alis.append(acc(pred, target).item())

        print(llis[-1], alis[-1])

    return [llis, alis]


@t.no_grad()
def validate_(net, val_loader, criterion,
              device=t.device('cuda' if t.cuda.is_available() else 'cpu'),
              ):
    acc = clfmet.Accuracy(top_k=1)
    llis, alis = list(), list()
    for imgs, target in val_loader:

        imgs = imgs.to(device=device)
        target = target.to(device=device)

        pred = net(imgs)
        loss = criterion(pred, target)

        llis.append(loss.item())
        alis.append(acc(pred, target).item())

        print(llis[-1], alis[-1])

    return [llis, alis]




