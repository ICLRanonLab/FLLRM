# -*- coding: utf8 -*

# system lib
import sys
import os

# third part libs
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

# my libs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import myData




def train_test_VAE(data, labels, net, device, optimizer, lossFunc, opt, pathName, fileName, saveModel=0):
    torch.manual_seed(16)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    zn, xn, yn = data.size()
    trainData, trainLabel, testData, testLabel = myData.separateData(labels, data, sep=5)
    dataSet = myData.MyDataset(trainData, trainLabel)
    dataLoader = DataLoader(dataset=dataSet,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.n_cpu,
                            pin_memory=torch.cuda.is_available())
    # train start
    for epoch in range(opt.n_epochs):
        c = 0
        loss_total = 0
        for step, (x, y) in enumerate(dataLoader):
            b_x = Variable(x.view(-1, xn * yn).float().to(device))  # batch data
            _, decoded, _ = net(b_x)
            loss = lossFunc(decoded, b_x)
            loss_total = loss_total + loss.data.cpu().numpy()
            c = c + 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
        loss_total = loss_total / c
        if epoch % 100 == 99:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss_total)
        scheduler.step(loss_total)
    # train end

    # save net
    if saveModel == 1:
        if not os.path.exists(pathName):
            os.makedirs(pathName)
        fileName = pathName + fileName
        torch.save(net, fileName)
    return (1)

def fit_VAE(data, labels, net, device, opt):
    zn, xn, yn = data.size()
    dataSet = myData.MyDataset(data, data)
    dataLoader = DataLoader(dataset=dataSet,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.n_cpu,
                            pin_memory=torch.cuda.is_available())
    # ----------------------------------------------------------fit-----------------------------------------
    # fit start
    ytrue_ypred = list()
    for (x, y) in dataLoader:
        b_x = Variable(x.view(-1, xn * yn).float().to(device))  # batch data
        b_y = Variable(y.to(device))  # batch y (label)
        _, _, predicted = net(b_x)
        predicted = torch.max(predicted.data, 1)[1].cpu()
        b_y = b_y.cpu()
        ytrue_ypred.append([b_y.numpy(), predicted.numpy()])
    return (ytrue_ypred)
