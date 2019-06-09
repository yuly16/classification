import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import argparse
import torch
import math
import random
from torch.utils.data import DataLoader
from pretrainedmodels import pretrainedmodels
from pretrainedmodels.pretrainedmodels import utils
import torchvision.transforms as transforms
from torchvision import models, transforms
from inceptionresnetv2 import *
import argparse
from train_inceptionv3 import inceptionv3
from utils import *
def arr2tag(arr1, arr2):
    tags = []
    index1 = np.where(arr1 > 0.6)
    index2 = np.where(arr2 > 0.6)
    arr = np.zeros(arr1.shape)
    arr[index1] = 1 
    arr[index2] = 1 
    return arr

def test(model1,model2,dataloader,device):
    global max_f1
    f1 = []
    pre = []
    rec = []
    total = 0
    model1.eval()
    model2.eval()
    with torch.no_grad():
        all_labels = ()
        all_pre = ()
        for imgs, labels in tqdm(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)                  
            y_pre1 = model1(imgs)
            y_pre2 = model2(imgs)
            y_pre = arr2tag(np.array(y_pre1.cpu()),np.array(y_pre2.cpu()))
            y_pre = torch.from_numpy(y_pre).float()
            for i in range(labels.shape[0]):
                all_pre += (y_pre[i,:],)
                all_labels += (labels[i,:].cpu(),)
        all_pre = torch.stack(all_pre,dim=0)
        all_labels = torch.stack(all_labels,dim=0)
        class_f1 = class_fbeta_score(all_labels,all_pre)
        class_prec = class_precision(all_labels,all_pre)
        class_rec = class_recall(all_labels,all_pre)

        mf1 = fbeta_score(all_labels,all_pre)
        mpre = precision(all_labels,all_pre)
        mrec = recall(all_labels,all_pre)
        wacc = wAcc(all_labels,all_pre)
        macc = mAcc(all_labels,all_pre)
        print("f1-score: %f  \nprecision: %f \nrecall: %f \nwacc:%f \nmacc:%f"%(mf1,mpre,mrec,wacc,macc))
        print("class f1-score: \n %s"%str(class_f1))
        print("class precision: \n %s"%str(class_prec))
        print("class recall: \n %s"%str(class_rec))
        return mf1

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--path',type=str,default='/home/yuly/multiclass/PascalVOC')
    parser.add_argument('--classes_num', type=int, default=20)
    opt=vars(parser.parse_args())
    tf_img = utils.TransformImage(pretrainedmodels.__dict__["inceptionv3"](num_classes=1000, pretrained='imagenet'))
    #cuda config
    use_cuda=True if torch.cuda.is_available() else False
    #use_cuda = False
    device=torch.device('cuda:0') if use_cuda else torch.device('cpu')
    torch.manual_seed(1)
    if use_cuda:
        torch.cuda.manual_seed(1)
    test_dataset = pretrainedmodels.datasets.Voc2007Classification(opt['path'], 'test', transform=tf_img)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=opt['batch_size'],shuffle=True)

    model1 = torch.load('inceptionresnetv2_best_model2.pth')
    model2 = torch.load('inceptionv3_best_model2.pth')
    model1.to(device)
    model2.to(device)
    test(model1,model2,test_dataloader,device)

if __name__ == "__main__":
    main()
