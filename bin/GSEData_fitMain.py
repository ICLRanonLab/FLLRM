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
parser.add_argument("--mean", type=int, default=0)
parser.add_argument("--errorStdBias", type=int, default=0)
parser.add_argument("--filterLen", type=int, default=7)
parser.add_argument("--n_filter", type=int, default=4)
parser.add_argument("--n_cluster", type=int, default=5)
parser.add_argument("--gseId", type=str, default="GSE72056") # gseId: GSE72056/GSE103322
runPams = parser.parse_args()


def getVAEPams(xn, yn, device, lr):
    VAE = myModules.VAE(xn, yn)
    VAE = VAE.to(device)
    optimizer = torch.optim.Adam(VAE.parameters(), lr=lr)
    lossFunc = nn.MSELoss()
    return (VAE, optimizer, lossFunc)


def main():
    # parameters
    gseId = runPams.gseId
    
    #the number of base 
    filterLen = runPams.filterLen

    # choose cpu or gpu automatically
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #torch.manual_seed(16)
    #data
    gseFile = "data/"+runPams.gseId+".csv" 
    gse = pd.read_csv(gseFile, index_col=0)
    gse = gse.iloc[0:100,0:100]
    gse_geneName = gse.index
    gse_cellName = gse.columns
    gse = gse.values.copy()
    
    
    # normalize the data by row
    gse = gse - (np.mean(gse, axis=1).reshape(gse.shape[0],1))
    gse_std = np.std(gse, axis=1).reshape(gse.shape[0],1)
    gse_std[gse_std==0] = 1
    gse = np.true_divide(gse, gse_std)
    mateData = pd.DataFrame(gse).copy()
    start_time = time.time()
    # slicing
    parts = myUtils.myData.mateData2Parts(mateData.copy())
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

        # normalize row and col by std from original 50*50's row and col std
        # rowFeatureMap = np.true_divide(rowFeatureMap, rowStdArr)

        #sort the col small -> big
        rowFeatureMap = -(np.sort(-(rowFeatureMap), axis=2))

        # normalize col by std from original 50*50' col
        # colFeatureMap = np.true_divide(colFeatureMap, colStdArr)
        # colFeatureMap = -np.sort(-colFeatureMap, axis=2)

        # resort them by their mean
        #delete the first 2 rows
        rowFeatureMap = myUtils.myData.getResortMeanFeatureMap(rowFeatureMap[:, :, 2:filterLen, :])
        # colFeatureMap = myUtils.myData.getResortMeanFeatureMap(colFeatureMap)

        # row and col max pooling 5*1652 -> 5*16
        rowFeatureMap = myUtils.myData.myMaxPooling(rowFeatureMap, baseTypeNumAfterKmean)
        # colFeatureMap = myUtils.myData.myMaxPooling(colFeatureMap, baseTypeNumAfterKmean)
        # featureMap = np.stack((rowFeatureMap, colFeatureMap), axis=2)

        # sort the rows small -> big
        rowFeatureMap = -(np.sort(-(rowFeatureMap), axis=3))[:, :, :, :]
        
        # denoising
        rowFeatureMap = rowFeatureMap[:, :, :, 0:filterLen] - rowFeatureMap[:, :, :, (16-filterLen):16]

        rowFeatureMap_np = rowFeatureMap.copy()
        labels = list()
        rowFeatureMap = torch.tensor(rowFeatureMap).float()
        rowFeatureMap = rowFeatureMap.view(rowFeatureMap.size()[0] * rowFeatureMap.size()[1], rowFeatureMap.size()[2],
                                           rowFeatureMap.size()[3])

        # optFeatureMap data
        fileName = "classifier.pkl"
        net = torch.load(fileName, map_location=device)
        net = net.to(device)
        
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
            tmpResults.append(pool.apply_async(myUtils.myData.getNewPart_clustering, args=(samples, mateData, runPams.n_cluster)))
        pool.close()
        pool.join()
        end_time = time.time()
        print("time:",end_time-start_time)
        # get new partitions
        newParts = list()
        for res in tmpResults:
            newParts.append(res.get())
        parts = newParts
        clusterRes = list()
        # update the index and col to geneName and cellName 
        for i1 in range(len(newParts)):
            for j1 in range(len(newParts[i1])):
                idx = newParts[i1][j1].index
                clmns = newParts[i1][j1].columns
                newParts[i1][j1].index = gse_geneName[idx]
                newParts[i1][j1].columns = gse_cellName[clmns]
                clusterRes.append(newParts[i1][j1])

        # save the clusters
        np.save(gseId+"_clusterRes.npy",clusterRes)
        print("end")
    return ()


# run main
if __name__ == "__main__":
    main()
