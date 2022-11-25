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





def train_(net, train_loaders, criterion, opt_fn, ustep,
           device=t.device('cuda' if t.cuda.is_available() else 'cpu'),
           ):
    acc = clfmet.Accuracy(top_k=1)
    llis, alis, samples = list(), list(), 0
    for train_loader in train_loaders:
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
    acc = clfmet.MulticlassAccuracy(num_classes=3)
    rec = clfmet.MulticlassRecall(num_classes=3)
    f1s = clfmet.MulticlassF1Score(num_classes=3)
    pre = clfmet.MulticlassPrecision(num_classes=3)
    spe = clfmet.MulticlassSpecificity(num_classes=3)
    llis, alis = list(), list() #loss and accuracy
    plis, slis = list(), list() # precision and specificity
    rlis, f1lis = list(), list() # recall and f1Score
    for imgs, target in val_loader:

        imgs = imgs.to(device=device)
        target = target.to(device=device)

        pred = net(imgs)
        loss = criterion(pred, target)

        llis.append(loss.item())
        alis.append(acc(pred, target).item())
        rlis.append(rec(pred, target).item())
        plis.append(pre(pred, target).item())
        slis.append(spe(pred, target).item())
        f1lis.append(f1s(pred, target).item())
        print("loss: ", llis[-1])
        print("Accuracy: ",alis[-1])
        print("recall: ", rlis[-1])
        print("Precision: ",plis[-1])
        print("specificity: ", slis[-1])
        print("F1Score: ",f1lis[-1] )
    return {"Loss": llis ,"Accuracy": alis,
            "recall":rlis, "Precision":plis,
            "specificity":slis, "F1Score":f1lis}



def kfoldTraining(net_class=network, loaders =None, profile =None,
                       epochs=50, startwith=0,
                       criterion=None, opt_fn=None, ustep = None,
                       device=t.device('cuda' if t.cuda.is_available() else 'cpu'),
                       ):
    """
        Profile shall maps the KfoldNumber,K, to epoch NUMber ,N, to
        { Train:{"Loss":list # list of all losess for each step,
                "Accuracy":list #list of accuracy for each step }
          Validation:{"Loss": llis ,"Accuracy": alis,
            "recall":rlis, "Precision":plis,
            "specificity":slis, "F1Score":f1lis}
    """

    for testfold in range(5):
        t.cuda.empty_cache()
        net = net_class()
        net.to(device=device)
        profile[testfold] = dict()
        for e in range(epochs):
            e += startwith
            profile[testfold][e]= {
                "Train":{"Loss":list(),
                         "Accuracy":list()},
                "Validate":None }#it WillBe assignedto directilly to the out of validate_
            # start training without loaderof NUMber testfold
            for i, loader in enumerate(loaders):
                if i == testfold :
                    continue
                l, a = train_(net, loader,criterion,
                              opt_fn, ustep, device)
                profile[e][testfold]["Train"]["Loss"] += l
                profile[e][testfold]["Train"]["Accuracy"] += a
            # traing finishedfor that fold and start validating
            profile[e][testfold]["Validate"] = validate_(
                net,loaders[testfold], criterion, device)



        net.train()
        tr_profile[e] += train_(net, trldr, criterion,
                                opt_fn, augm, ustep,
                                device)

        save_model(net, e)

        net.eval()
        tr_profile[e] += validate_(net, valdr, criterion,
                                   augm, device)
        save_train_hist(tr_profile, e)
        save_train_hist(aug.transforms[si].ScaleHist.tolist(),
                        e, name='augfr')

    return tr_profile

