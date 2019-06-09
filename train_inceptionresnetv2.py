import argparse
import torch
import numpy as np
import math
import random
from torch.utils.data import DataLoader
from pretrainedmodels import pretrainedmodels
from pretrainedmodels.pretrainedmodels import utils
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import os
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from inceptionresnetv2 import *
import numpy as np
#import cv2
import json
from tqdm import tqdm
#plt.switch_backend('agg')
# %matplotlib inline

from glob import glob
from tqdm import tqdm
from utils import *
#import cv2
from PIL import Image
import argparse
import torch
import numpy as np
import math

# from torch.utils.data import DataLoader


global net
def test(net,dataloader,device):
    global max_f1
    # class_prec = np.zeros((20))
    # class_rec = np.zeros((20))
    # class_f1 = np.zeros((20))
    f1 = []
    pre = []
    rec = []
    total = 0
    net.eval()
    with torch.no_grad():
        all_labels = ()
        all_pre = ()
        for imgs, labels in tqdm(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)                  
            y_pre = net(imgs)
            y_pre += threshold(thre)
            for i in range(labels.shape[0]):
                all_pre += (y_pre[i,:],)
                all_labels += (labels[i,:],)
            # f1.append(np.array(fmeasure(labels,y_pre).cpu()))
            # pre.append(np.array(precision(labels,y_pre).cpu()))
            # rec.append(np.array(recall(labels,y_pre).cpu()))

        
        all_pre = torch.stack(all_pre,dim=0)
        all_labels = torch.stack(all_labels,dim=0)
        class_f1 = class_fbeta_score(all_labels,all_pre)
        class_prec = class_precision(all_labels,all_pre)
        class_rec = class_recall(all_labels,all_pre)
        # mf1 = np.mean(f1)
        # mpre = np.mean(pre)
        # mrec = np.mean(rec)
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
    parser.add_argument('--epoch1', type=int, default=20)
    parser.add_argument('--epoch2', type=int, default=8)
    parser.add_argument('--image_size', default=(299,299))
    parser.add_argument('--path',type=str,default='/home/yuly/multiclass/PascalVOC')
    parser.add_argument('--classes_num', type=int, default=20)
    parser.add_argument('--stage', type=int, default=0)
    opt=vars(parser.parse_args())
    tf_img = utils.TransformImage(pretrainedmodels.__dict__["inceptionv3"](num_classes=1000, pretrained='imagenet'))

    #cuda config
    use_cuda=True if torch.cuda.is_available() else False
    #use_cuda = False
    device=torch.device('cuda:0') if use_cuda else torch.device('cpu')
    torch.manual_seed(1)
    if use_cuda:
        torch.cuda.manual_seed(1)
    # img_transform = transforms.Compose([transforms.ToTensor(),
                                        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    train_dataset = pretrainedmodels.datasets.Voc2007Classification(opt['path'], 'train', transform=tf_img)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=opt['batch_size'],shuffle=True)
    test_dataset = pretrainedmodels.datasets.Voc2007Classification(opt['path'], 'test', transform=tf_img)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=opt['batch_size'],shuffle=True)
    if opt['stage'] == 0:
        net = inceptionresnetv2(num_classes=20, pretrained='imagenet')
        net.to(device)
    elif opt['stage'] == 1:
        print('we will load model')      
    elif opt['stage'] == 2:
        print('Test procedure')  

    weight = torch.Tensor([2465. , 463. , 756. , 232.,  908.,  284.,  261. , 435. , 369.  ,302. , 325. , 735., 325. , 398. , 492.,  868. , 449.,  368.,  478.,  373.])
    weight = 1./weight
    weight = 20*weight/sum(weight)

    if opt['stage'] <= 0:
        optimizer1 = torch.optim.Adam(list(net.last_linear1.parameters())+list(net.old_module.last_linear.parameters()),lr=1e-3)
        max_f1 = 0
        for epoch in range(opt['epoch1']):
            net.train()
            print('training procedure:')
            print(len(train_dataloader))
            for imgs, labels in tqdm(train_dataloader):
                imgs = imgs.to(device)
                labels = labels.to(device)
                optimizer1.zero_grad()
                criterion = torch.nn.BCELoss(weight=weight.to(device))
                y_pre = net(imgs)
                #print(y_pre.shape)
                BCE_loss = criterion(y_pre,labels)
                BCE_loss.backward()
                optimizer1.step()
            print("loss %f the %d step in total %d epochs finished"%(BCE_loss,epoch,opt['epoch1']))
            # # test
            print('\n')
            print('testing procedure:')
            mf1 = test(net,test_dataloader,device)
            if mf1>max_f1:
                torch.save(net,'inceptionresnetv2_best_model1.pth')
                print('best model, saving…')
                max_f1 = mf1


    if opt['stage'] <= 1:
        net = torch.load('inceptionresnetv2_best_model1.pth')
        net.to(device)

        # training step2
        optimizer2=torch.optim.Adam([{'params': net.old_module.mixed_5b.parameters(),'lr':1e-4},
            {'params': net.old_module.repeat.parameters(),'lr':1e-4},
            {'params': net.old_module.mixed_6a.parameters(),'lr':1e-4},
            {'params': net.old_module.repeat_1.parameters(),'lr':1e-4},
            {'params': net.old_module.mixed_7a.parameters(),'lr':1e-4},
            {'params': net.old_module.repeat_2.parameters(),'lr':1e-4},
            {'params': net.old_module.block8.parameters(),'lr':1e-4},
            {'params': net.old_module.conv2d_7b.parameters(),'lr':1e-4},
            {'params':net.last_linear1.parameters(),'lr':1e-4},
            {'params':net.old_module.last_linear.parameters(),'lr':1e-4}])
        max_f1 = 0 

        for epoch in range(opt['epoch1']):
            net.train()
            print('\n')
            print('\n')
            print('training procedure:')
            for imgs, labels in tqdm(train_dataloader):
                imgs = imgs.to(device)
                labels = labels.to(device)
                optimizer2.zero_grad()
                criterion = torch.nn.BCELoss(weight=weight.to(device))
                y_pre = net(imgs)
                #print(y_pre.shape)
                BCE_loss = criterion(y_pre,labels)
                BCE_loss.backward()
                optimizer2.step()
            print("loss %f the %d step in total %d epochs finished"%(BCE_loss,epoch,opt['epoch1']))
            # test
            print('\n')
            print('testing procedure:')
            mf1 = test(net,test_dataloader,device)
            if mf1>max_f1:
                torch.save(net,'inceptionresnetv2_best_model2.pth')
                print('best model, saving…')
                max_f1 = mf1


    if opt['stage'] <= 2:
        net = torch.load('inceptionresnetv2_best_model2.pth')
        net.to(device)
        _ = test(net,test_dataloader,device)
if __name__ == "__main__":
    main()
