# -*- coding: utf8 -*

# system lib
import argparse
import sys
import time
import os
from multiprocessing import Pool

# third part lib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# my libs
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath("../../"))

import Predicte.myModules
import myUtils.myTrainTest
import myUtils.myData

# parameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_inner_epochs", type=int, default=1)
if torch.cuda.is_available():
    parser.add_argument("--batch_size", type=int, default=512)
else:
    parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--n_cpu", type=int, default=os.cpu_count())
parser.add_argument("--xn", type=int, default=500)
parser.add_argument("--crType", type=str, default="uniform")
parser.add_argument("--mean", type=int, default=0)
parser.add_argument("--errorStdBias", type=int, default=0)
parser.add_argument("--testType", type=str, default="err")
parser.add_argument("--filterLen", type=int, default=7)
parser.add_argument("--n_filter", type=int, default=4)
runPams = parser.parse_args()

def main():
    # parameters
    yn = runPams.xn
    totalRow = 1000
    totalCol = 1000
    LK=1
    #get the true pattern and std of the pattern
    c = np.random.uniform(low=0.0, high=1.0, size=(runPams.xn,LK))
    r = np.random.uniform(low=0.0, high=1.0, size=(LK, yn))
    part = np.matmul(c,r)
    part_std = np.std(part)
    
    #the number of base 
    filterLen = runPams.filterLen

    # error & mean
    runPams.errorStdBias = runPams.errorStdBias / 10
    runPams.errorStdBias = runPams.errorStdBias * (part_std)
    runPams.mean = runPams.mean / 10
    runPams.mean = runPams.mean * (part_std)

    # choose cpu or gpu automatically
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # back ground noise: normal distribution mean:0, sigma: the sd of true pattern
    bgNoise = np.random.normal(loc=0.0, scale=part_std, size=(totalRow, totalCol))
    # normalize the data 
    bgNoise = bgNoise - (np.mean(bgNoise, axis=1).reshape(bgNoise.shape[0],1))
    bgNoise_std = np.std(bgNoise, axis=1).reshape(bgNoise.shape[0],1)
    bgNoise_std[bgNoise_std==0] = 1
    bgNoise = np.true_divide(bgNoise, bgNoise_std)


    # normalize the true pattern by row 
    part = part - (np.mean(part, axis=1).reshape(part.shape[0],1))
    part_std = np.std(part, axis=1).reshape(part.shape[0],1)
    part_std[part_std==0]=1
    part = np.true_divide(part, part_std)

    error = np.random.normal(loc=0, scale=runPams.errorStdBias, size=(runPams.xn, yn))
    part = part + error + runPams.mean 
    
    # replace the bgNoise 
    bgNoise[0:runPams.xn,0:yn] = part 
    # shuffle the data
    mateData = pd.DataFrame(bgNoise).copy()
    mateData = shuffle(mateData)
    mateData = shuffle(mateData.T)
    mateData = mateData.T


    # slicing
    parts = myUtils.myData.mateData2Parts(mateData.copy())
    newParts = list() 
    for _ in range(runPams.n_inner_epochs):
        res = list()
        pool = Pool(os.cpu_count())
        for part in parts:
            res.append(pool.apply_async(myUtils.myData.getSamplesRowColStd, args=(part, filterLen)))
        pool.close()
        pool.join()

        # splite samples, samplesArr, rowStdArr, colStdArr from res
        samples = list()
        samplesArr = list()
        # rowStdArr = list()
        # colStdArr = list()
        for r in res:
            tmp = r.get()
            samples.append(tmp[0])
            samplesArr.append(tmp[1])
            # rowStdArr.append(r[2])
            # colStdArr.append(r[3])
        samplesArr = np.stack(samplesArr)
        
        # rowStdArr = np.stack(rowStdArr)
        # colStdArr = np.stack(colStdArr)

        # get bases matrix
        basesMtrx, baseTypeNumAfterKmean, baseIdAfterKMeans = myUtils.myData.getBasesMtrxAfterKmean_mul(filterLen,n_filter=runPams.n_filter) 
        
        # get row and col feature map:
        rowFeatureMap = np.matmul(samplesArr, (basesMtrx.iloc[:, 0:filterLen].values.T))
        # colFeatureMap = np.matmul(samplesArr.transpose((0, 1, 3, 2)), (basesMtrx.iloc[:, 0:7].values.T))


        #sort the col small -> big
        rowFeatureMap = -(np.sort(-(rowFeatureMap), axis=2))

        # normalize col by std from original 50*50' col
        # colFeatureMap = np.true_divide(colFeatureMap, colStdArr)
        # colFeatureMap = -np.sort(-colFeatureMap, axis=2)

        # resort them by their mean
        #delete the first 2 rows
        rowFeatureMap = myUtils.myData.getResortMeanFeatureMap(rowFeatureMap[:, :, 2:filterLen, :])
        # colFeatureMap = myUtils.myData.getResortMeanFeatureMap(colFeatureMap)

        # row and col max pooling
        rowFeatureMap = myUtils.myData.myMaxPooling(rowFeatureMap, baseTypeNumAfterKmean)
        # colFeatureMap = myUtils.myData.myMaxPooling(colFeatureMap, baseTypeNumAfterKmean)
        # featureMap = np.stack((rowFeatureMap, colFeatureMap), axis=2)

        # sort the rows small -> big
        rowFeatureMap = -(np.sort(-(rowFeatureMap), axis=3))[:, :, :, :]
        
        rowFeatureMap = rowFeatureMap[:, :, :, 0:filterLen] - rowFeatureMap[:, :, :, (16-filterLen):16]

        #rowFeatureMap_np = rowFeatureMap.copy()
        labels = list() 
        rowFeatureMap = torch.tensor(rowFeatureMap).float()
        rowFeatureMap = rowFeatureMap.view(rowFeatureMap.size()[0] * rowFeatureMap.size()[1], rowFeatureMap.size()[2],
                                           rowFeatureMap.size()[3])

        # optFeatureMap data
        pathName = ""
        fileName = pathName + "classifier.pkl"
        net = torch.load(fileName,map_location=device)
        net = net.to(device)
        #net.eval()
        
        
        predLabels = myUtils.myTrainTest.fit_VAE(rowFeatureMap, labels, net, device, runPams)
        
        predLabels = [pl[1] for pl in predLabels]
        predLabels = np.concatenate(predLabels)
        

        if (len(predLabels) != (len(samples)*len(samples[0]))):
            print("size match error!")
            return()
        
        labelType =np.sort(np.unique(predLabels))
        classNum = len(labelType)
        predLabels = np.resize(predLabels, (len(samples), len(samples[0]), 1))

        # get update row and col indices
        # initial the new empty samples list
        allNewSamples = list()
        for _ in range(classNum):
            allNewSamples.append([])

        # re generate the samples by their generated label
        sampleSetNum = len(samples)
        samplesNum = len(samples[0])
        for i in range(sampleSetNum):
            for j in range(samplesNum):
                label = predLabels[i][j]
                idx = np.where(labelType == label.item())[0][0]
                allNewSamples[idx].append(samples[i][j])

        # get new expand samples from mateData
        pool = Pool(os.cpu_count())
        tmpResults = list()
        for samples in allNewSamples:
            tmpResults.append(pool.apply_async(myUtils.myData.getNewPart, args=(samples, mateData, runPams.xn)))
        pool.close()
        pool.join()

        # get new partitions
        newParts = list()
        for res in tmpResults:
            newParts.append(res.get())
        parts = newParts
    
    # caculate the match degree
    matchLabel = list()
    sigmaRatio = list()
    newParts = parts
    for newPart in newParts:
        if len(newPart) == 0:
            matchLabel.append("Nan")
            continue
        matchRowLen = np.sum(list(map(lambda x: x < runPams.xn, newPart.index)))
        matchColLen = np.sum(list(map(lambda x: x < yn, newPart.columns)))
        accuracy = ((matchRowLen * matchColLen) / (yn * yn)) * 100
        accuracy = np.around(accuracy, decimals=2)
        matchLabel.append(accuracy)
    
    acc_p = np.max(matchLabel)
    acc_n = 100 - np.min(matchLabel)

    # output the results
    res = list()
    res.append(str(runPams.xn))
    res.append(str(np.around(runPams.mean, decimals=2)))
    res.append(str(np.around(runPams.errorStdBias, decimals=2)))
    res.append(str(np.around(acc_p, 2)))
    res.append(str(np.around(acc_n, 2)))
    res = pd.DataFrame(res)
    res = res.T
    print(res)
    print("end")
    return ()


# run main
if __name__ == "__main__":
    main()
