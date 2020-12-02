# -*- coding: utf-8 -*

# sys libs
import sys
import time

# 3rd libs
import torch
import numpy as np
import pandas as pd
from numba import jit
import pysnooper
from functools import lru_cache
# my libs

# gene dataset loader
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    # end

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    # end

    def __len__(self):
        return len(self.data)

    # end


# end
@jit(nopython=True)
def getMaxIdx_jit(corectEf, num):
    maxIdx = list()
    rowNum = corectEf.shape[0]
    colNum = corectEf.shape[1]
    while True:
        if len(set(maxIdx)) >= num:
            break
        max = -10000
        r = 0
        c = 0
        for i in range(rowNum - 1):
            for j in range(i + 1, colNum):
                if corectEf[i][j] > max:
                    max = corectEf[i][j]
                    r = i
                    c = j
        maxIdx.append(r)
        maxIdx.append(c)
        corectEf[r][c] = -1000
    maxIdx = sorted(list(set(maxIdx))[:num])
    return (maxIdx)


# end


@jit(nopython=True)
def getMaxIdx_jit_new(corectEf, num):
    maxIdx = list()
    rowNum = corectEf.shape[0]
    colNum = corectEf.shape[1]
    for _ in range(num * 2):
        if len(set(maxIdx)) >= num:
            break
        max = 0
        r = 0
        c = 0
        for i in range(rowNum - 1):
            for j in range(i + 1, colNum):
                if corectEf[i][j] > max:
                    max = corectEf[i][j]
                    r = i
                    c = j
        maxIdx.append(r)
        maxIdx.append(c)
        corectEf[r][c] = -1000
    maxIdx = sorted(list(set(maxIdx))[:num])
    return (maxIdx)


def getMaxIdx(corectEf, num):
    maxIdx = list()
    while len(maxIdx) < num:
        mi = np.where(corectEf == corectEf.max())
        corectEf[mi] = -1000
        maxIdx = np.append(mi[0], mi[1])
        maxIdx = np.unique(maxIdx)
    maxIdx = maxIdx[:num]
    return (maxIdx)


# end


def getMaxIdx_new(corectEf, num):
    value = np.argpartition(corectEf.ravel(), -num)[-num:]
    i2d = np.unravel_index(value, corectEf.shape)
    idx = np.append(i2d[0], i2d[1])
    idx = np.unique(idx)
    return (idx)


# end


def getMaxIdx_bigSize(corectEf, num):
    # corectEf = torch.tensor(corectEf)
    col_max_values = np.max(corectEf, axis=0)
    col_max_idx = col_max_values.argsort()[::-1][0:num]
    # idx = torch.max(corectEf, dim=0)[0]
    # idx = torch.max(a)[1]
    # values, _ = corectEf.topk(1, dim=0, largest=True, sorted=True)
    # _, idx = values.topk(num)
    return (col_max_idx)


# end


@jit(nopython=True)
def addNantoEf(corectEf):
    rowNum = corectEf.shape[0]
    colNum = corectEf.shape[1]
    for i in range(colNum):
        for j in range(i, rowNum):
            corectEf[j][i] = np.nan
    return (corectEf)


# end


@jit(nopython=True)
def getLabelFrRCIdx(rowIdx, colIdx, patternLen):
    rn = 0
    cn = 0
    for i in rowIdx:
        if i < patternLen:
            rn += 1
    for i in colIdx:
        if i < patternLen:
            cn += 1
    return (rn * cn)


@jit(nopython=True)
def getcorrcoef(mtrx):
    return (np.corrcoef(mtrx))


# end



def mateData2Parts_realData(mateData):
    partRowLen = 50
    patternLen = partRowLen
    partColLen = partRowLen
    xn, yn = mateData.shape
    loopNum = yn if xn > yn else xn
    # loopNum = int(len(mateData) / partRowLen)
    sizeNum = [5, 10, 20, 50]
    parts = list()
    for i in range(loopNum):
        if mateData.shape[0] == partRowLen or mateData.shape[1] == partColLen:
            part = mateData
            part = part - np.mean(part)
            rowStd = np.std(part, axis=1)
            colStd = np.std(part, axis=0)
            # part.loc[:, "rowStd"] = rowStd
            # part.loc["colStd", :] = colStd
            parts.append(part)
            break

        # get the part row and col idx
        randomInitRowIdx = np.random.randint(len(mateData), size=sizeNum[0])

        initRowIdx = randomInitRowIdx
        partRowIdx = list()
        partColIdx = list()
        for sn in sizeNum:
            colCorectEf = np.corrcoef(mateData.iloc[initRowIdx, :].T)
            colCorectEf[np.tril_indices(colCorectEf.shape[0], 0)] = -1000
            colMaxIdx = getMaxIdx_jit(colCorectEf.copy(), sn)
            rowCorectEf = np.corrcoef(mateData.iloc[:, colMaxIdx])
            rowCorectEf[np.tril_indices(rowCorectEf.shape[0], 0)] = -1000
            rowMaxIdx = getMaxIdx_jit(rowCorectEf.copy(), sn)
            initRowIdx = rowMaxIdx
            partRowIdx = rowMaxIdx
            partColIdx = colMaxIdx

        # get one part from mateData by row and col idx.
        part = mateData.iloc[partRowIdx, partColIdx].copy()
        mateData = mateData.drop(index=part.index)
        mateData = mateData.drop(columns=part.columns)

        # part minus row mean value
        part = part - (np.mean(part.values, axis=1).reshape(patternLen, 1))
        std = np.std(part.values, axis=1).reshape(partRowLen, 1)
        std[std == 0.0] = 1.0
        part = np.true_divide(part, std)
        # get row std and col std
        rowStd = np.std(part, axis=1)
        colStd = np.std(part, axis=0)
        # part.loc[:, "rowStd"] = rowStd
        # part.loc["colStd", :] = colStd
        parts.append(part)
    return (parts)


#end

def mateData2Parts(mateData):
    partRowLen = 50
    patternLen = partRowLen
    partColLen = partRowLen
    xn, yn = mateData.shape
    loopNum = yn if xn > yn else xn
    # loopNum = int(len(mateData) / partRowLen)
    sizeNum = [5, 10, 20, 50]
    parts = list()
    for i in range(loopNum):
        if mateData.shape[0] == partRowLen or mateData.shape[1] == partColLen:
            part = mateData
            part = part - np.mean(part)
            rowStd = np.std(part, axis=1)
            colStd = np.std(part, axis=0)
            # part.loc[:, "rowStd"] = rowStd
            # part.loc["colStd", :] = colStd
            parts.append(part)
            break

        # get the part row and col idx
        randomInitRowIdx = np.random.randint(len(mateData), size=sizeNum[0])
        initRowIdx = randomInitRowIdx
        partRowIdx = list()
        partColIdx = list()
        for sn in sizeNum:
            colCorectEf = np.corrcoef(mateData.iloc[initRowIdx, :].T)
            colCorectEf[np.tril_indices(colCorectEf.shape[0], 0)] = -1000
            colMaxIdx = getMaxIdx_jit(colCorectEf.copy(), sn)
            rowCorectEf = np.corrcoef(mateData.iloc[:, colMaxIdx])
            rowCorectEf[np.tril_indices(rowCorectEf.shape[0], 0)] = -1000
            rowMaxIdx = getMaxIdx_jit(rowCorectEf.copy(), sn)
            initRowIdx = rowMaxIdx
            partRowIdx = rowMaxIdx
            partColIdx = colMaxIdx

        # get one part from mateData by row and col idx.
        part = mateData.iloc[partRowIdx, partColIdx].copy()
        mateData = mateData.drop(index=part.index)
        mateData = mateData.drop(columns=part.columns)

        # part minus row mean value
        part = part - (np.mean(part.values, axis=1).reshape(patternLen, 1))
        std = np.std(part.values, axis=1).reshape(partRowLen, 1)
        std[std == 0.0] = 1.0
        part = np.true_divide(part, std)
        # get row std and col std
        rowStd = np.std(part, axis=1)
        colStd = np.std(part, axis=0)
        # part.loc[:, "rowStd"] = rowStd
        # part.loc["colStd", :] = colStd
        parts.append(part)
    return (parts)


#end

def getR2_colBase(part, n_maxCol):
    n_row, n_col = part.shape
    sigmas = list()
    for j in range(n_col):
        _,sigma,_ = np.linalg.svd(part.iloc[:,j].values.reshape(n_row,1))
        sigmas.append(sigma[0])
    maxIdx = np.argpartition(sigmas, -n_maxCol)[-n_maxCol:]
    part = part.iloc[:,maxIdx]
    colNames = part.columns

    colR2 = np.corrcoef(part.T)
    #colR2[np.tril_indices(colR2.shape[0], 0)] = 0
    colR2 = colR2**2
    r2_mean = np.mean(colR2)
    return(r2_mean, colNames)

def getR2_rowBase(part, n_maxRow):
    n_row, n_col = part.shape
    sigmas = list()
    for i in range(n_row):
        _,sigma,_ = np.linalg.svd(part.iloc[i,:].values.reshape(n_col,1))
        sigmas.append(sigma[0])
    maxIdx = np.argpartition(sigmas, -n_maxRow)[-n_maxRow:]
    part = part.iloc[maxIdx,:]
    colNames = part.columns

    colR2 = np.corrcoef(part)
    #colR2[np.tril_indices(colR2.shape[0], 0)] = 0
    colR2 = colR2**2
    r2_mean = np.mean(colR2)
    return(r2_mean, colNames)

# matrix slicing
def mateData2Parts_r2(mateData):
    partRowLen = 50
    patternLen = partRowLen
    partColLen = partRowLen
    xn, yn = mateData.shape
    loopNum = yn if xn > yn else xn
    # loopNum = int(len(mateData) / partRowLen)
    
    # loop parameters 
    s_MS = [5, 10, 25, 50,50]
    r_MS = [1000,100,10,1,1]

    parts = list()
    for _ in range(loopNum):
        if mateData.shape[0] == partRowLen or mateData.shape[1] == partColLen:
            part = mateData
            part = part - np.mean(part)
            rowStd = np.std(part, axis=1)
            colStd = np.std(part, axis=0)
            # part.loc[:, "rowStd"] = rowStd
            # part.loc["colStd", :] = colStd
            parts.append(part)
            break

        partRowIdx = list()
        partColIdx = list()
            
        st = time.time()
        # get init parts randomly
        initParts=list() 
        for _ in range(r_MS[0]):
            rdmIdx = np.random.randint(len(mateData), size=s_MS[0])
            part = mateData.iloc[rdmIdx,:]
            initParts.append(part)
        
        # get new part by iteration 
        for i in range(1,len(s_MS)-1):
            # get new col sets
            r2MeanScores = list()
            colIdxSets = list()
            for part in initParts:
                r2_mean, colIdxSet = getR2_colBase(part, s_MS[i])
                r2MeanScores.append(r2_mean)
                colIdxSets.append(colIdxSet)

            maxIdx = np.argpartition(r2MeanScores, -r_MS[i])[-r_MS[i]:]
            colIdxSets = [ colIdxSets[mi] for mi in maxIdx ]# get new col index sets after computing sigmas and r2
            
            initParts = list()
            for idx in colIdxSets:
                part = mateData.iloc[:,idx]
                initParts.append(part)
            
            # get new row sets 
            r2MeanScores = list()
            rowIdxSets = list()
            for part in initParts:
                r2_mean, rowIdxSet = getR2_rowBase(part, s_MS[i+1])
                r2MeanScores.append(r2_mean)
                rowIdxSets.append(rowIdxSet)

            maxIdx = np.argpartition(r2MeanScores, -r_MS[i+1])[-r_MS[i+1]:]
            rowIdxSets = [ rowIdxSets[mi] for mi in maxIdx ]

            initParts = list() 
            for idx in rowIdxSets:
                part = mateData.iloc[idx,:]
                initParts.append(part)
            
            print(initParts[0])
            print(initParts[0].shape)
            print(len(initParts))
            et = time.time()-st
            print(et)
            sys.exit()
            # get final row idx and col idx
            if s_MS[i+1] == s_MS[-1]:
                partRowIdx = initParts[0].index
                partColIdx = initParts[0].columns
        # get one part from mateData by row and col idx.
        part = mateData.iloc[partRowIdx, partColIdx].copy()
        mateData = mateData.drop(index=part.index)
        mateData = mateData.drop(columns=part.columns)

        # part minus row mean value
        part = part - (np.mean(part.values, axis=1).reshape(patternLen, 1))
        std = np.std(part.values, axis=1).reshape(partRowLen, 1)
        std[std == 0.0] = 1.0
        part = np.true_divide(part, std)
        # get row std and col std
        rowStd = np.std(part, axis=1)
        colStd = np.std(part, axis=0)
        # part.loc[:, "rowStd"] = rowStd
        # part.loc["colStd", :] = colStd
        parts.append(part)
    return (parts)

#end

def generatePart(patternType="uniform",
                 LK=1,
                 partRow=50,
                 partCol=50):
    print(LK)
    parts = list()
    # uniform distribution
    if patternType == "uniform":
        # add error
        for i in range(0,21,2):
            for _ in range(1, 5):
                #np.random.seed(j)
                c = np.random.uniform(low=0.0, high=1.0, size=(partRow, LK))
                r = np.random.uniform(low=0.0, high=1.0, size=(LK, partCol))
                part = np.matmul(c, r)
                part_std = np.std(part)
                err = np.random.normal(loc=0.0, scale=((i/10)*part_std), size=(partRow, partCol))
                part = part - (np.mean(part, axis=1).reshape(partRow, 1))
                std = np.std(part, axis=1).reshape(partRow, 1)
                std[std == 0.0] = 1
                part = np.true_divide(part, std)
                part = part + err
                parts.append(part)

    # norm distributon
    if patternType == "bgNoise":
        for i in range(1):
            for _ in range(1, 5):
                #np.random.seed(j)
                c = np.random.uniform(low=0.0, high=1.0, size=(partRow, LK))
                r = np.random.uniform(low=0.0, high=1.0, size=(LK, partCol))
                part = np.matmul(c, r)
                part_std = np.std(part)
                bgNoise = np.random.normal(loc=0.0, scale=part_std, size=(partRow, partCol))
                #part = part - np.mean(part)
                parts.append(bgNoise)
    return (parts)


#end

def generatePart_old(patternType="uniform",
                 LK=1,
                 partRow=50,
                 partCol=50,
                 errNum=50):
    parts = list()
    # uniform distribution
    if patternType == "uniform":
        for i in range(0, errNum + 1, 5):
            err = i / 100
            err = np.random.normal(0, err, (partRow, partCol))
            c = np.random.rand(partRow, LK)
            r = np.random.rand(LK, partCol)
            part = np.matmul(c, r) + err
            part = part - (np.mean(part, axis=1).reshape(partRow, 1))
            std = np.std(part, axis=1).reshape(partRow, 1)
            std[std == 0.0] = 1
            part = np.true_divide(part, std)
            parts.append(part)
    # norm distributon
    if patternType == "norm":
        for i in range(1):
            #err = i / 100
            part = np.random.randn(partRow, partCol)
            part = part - (np.mean(part, axis=1).reshape(partRow, 1))
            #std = np.std(part, axis=1).reshape(partRow, 1)
            #std[std == 0.0]=1
            #part = np.true_divide(part, std)
            parts.append(part)
    return (parts)


#end


def getSamples(part, baseLen, sampleNum = 500):
    randomRowColIdx = getRandomRowColIdx(low=0,
                                         high=49,
                                         row=2,
                                         col=baseLen,
                                         number=sampleNum)
    samplesArr = list()
    for rowIdx, colIdx in randomRowColIdx:
        sample = part[rowIdx].copy()
        sample = sample[:,colIdx].copy()
        samplesArr.append(sample)
        # rowStds.append(np.concatenate(part[["rowStd"]].iloc[rowIdx].values))
        # colStds.append(np.concatenate(part.T[["colStd"]].iloc[colIdx].values))
    samplesArr = np.stack(samplesArr)
    # rowStds = np.stack(rowStds).reshape(sampleNum, 7, 1)
    # colStds = np.stack(colStds).reshape(sampleNum, 7, 1)
    return (samplesArr)


# end


def getSamplesRowColStd(part, baseLen=5, sampleNum=500):
    randomRowColIdx = getRandomRowColIdx(low=0,
                                         high=49,
                                         row=2,
                                         col=baseLen,
                                         number=sampleNum)
    samples = list()
    samplesArr = list()
    rowStds = list()
    colStds = list()
    for rowIdx, colIdx in randomRowColIdx:
        sample = part.iloc[rowIdx, colIdx].copy()
        samples.append(sample)
        samplesArr.append(sample.values)
        # rowStds.append(np.concatenate(part[["rowStd"]].iloc[rowIdx].values))
        # colStds.append(np.concatenate(part.T[["colStd"]].iloc[colIdx].values))
    samplesArr = np.stack(samplesArr)
    # rowStds = np.stack(rowStds).reshape(sampleNum, 7, 1)
    # colStds = np.stack(colStds).reshape(sampleNum, 7, 1)

    return ([samples, samplesArr, rowStds, colStds])


# end


# @jit(nopython=True)
def partsMinusMeanGetStd(parts_rowIdx_colIdx):
    for p in range(len(parts_rowIdx_colIdx)):
        parts_rowIdx_colIdx[p][0] = parts_rowIdx_colIdx[p][0] - \
            np.mean(parts_rowIdx_colIdx[p][0])
        rowStd = np.std(parts_rowIdx_colIdx[p][0], axis=1)
        colStd = np.std(parts_rowIdx_colIdx[p][0], axis=0)
        parts_rowIdx_colIdx[p].append(rowStd)
        parts_rowIdx_colIdx[p].append(colStd)
    parts_rowIdx_colIdx_rowStd_colStd = parts_rowIdx_colIdx
    return (parts_rowIdx_colIdx_rowStd_colStd)


# end


def partsMinusRowMeanGetStd(parts_rowIdx_colIdx):
    partLen = parts_rowIdx_colIdx[0][0].shape[0]
    for p in range(len(parts_rowIdx_colIdx)):
        parts_rowIdx_colIdx[p][0] = parts_rowIdx_colIdx[p][0] - (np.mean(
            parts_rowIdx_colIdx[p][0], axis=1).reshape(partLen, 1))
        rowStd = np.std(parts_rowIdx_colIdx[p][0], axis=1)
        colStd = np.std(parts_rowIdx_colIdx[p][0], axis=0)
        parts_rowIdx_colIdx[p].append(rowStd)
        parts_rowIdx_colIdx[p].append(colStd)
        parts_rowIdx_colIdx_rowStd_colStd = parts_rowIdx_colIdx
    return (parts_rowIdx_colIdx_rowStd_colStd)


# end
def kMeans(x, k=32):
    from sklearn.cluster import KMeans
    x = x.T
    kmeansRes = KMeans.fit(x, k)
    # Save the labels
    x.loc[:, 'labels'] = kmeansRes.labels_
    return (x)


# end


class Part23dMap():
    def __init__(self, baseTypeNum, bases, totalRow, totalCol,
                 randomRowColIdx):
        self.baseTypeNum = baseTypeNum
        self.bases = bases
        self.totalRow = totalRow
        self.totalCol = totalCol
        self.randomRowColIdx = randomRowColIdx

    # end

    def getMaxSlice(self, baseFeature, startIdx, endIdx):
        maxSlice = torch.max(baseFeature[:, :, startIdx:endIdx], 2)
        maxSlice = maxSlice[0]  # get values
        z = maxSlice.size()[0]
        x = maxSlice.size()[1]
        y = 1
        maxSlice = maxSlice.reshape(z, x, y)
        return (maxSlice)

    # end

    def getBaseMap(self, samples):
        rowFeature = samples * self.bases.float()
        colFeature = samples.transpose(0, 2, 1) * self.bases.float()
        baseFeature = torch.cat((rowFeature, colFeature), 0)  # 1000x7xm
        # max pooling
        finalMap = [
            self.getMaxSlice(baseFeature, self.baseTypeNum[i],
                             self.baseTypeNum[i + 1])
            for i in range(len(self.baseTypeNum))
            if i < len(self.baseTypeNum) - 1
        ]
        finalMap = torch.cat(finalMap, 2)  # 1000x7xlen(bases)
        return (finalMap)

    # end

    def getRowColFeatureMapNoMaxPooling(self, samples):
        rowFeatureMap = np.matmul(samples, self.bases.astype(np.float64))
        colFeatureMap = np.matmul(samples.transpose(0, 2, 1),
                                  self.bases.astype(np.float64))
        # rowColFeatureMap = torch.stack((rowFeatureMap, colFeatureMap),1)  #
        # 2*500*7*M->500*2*7*M
        return (rowFeatureMap, colFeatureMap)

    # end

    def getSamples(self, partition):
        samples = [
            partition[np.ix_(rowIdx, colIdx)]
            for rowIdx, colIdx in self.randomRowColIdx
        ]
        samples = np.stack(samples)
        return (samples)

    # end

    def getFeatureMap(self, partition):
        samples = self.getSamples(partition)
        baseMap = self.getBaseMap(samples)
        featureMap = baseMap.permute(2, 1, 0)
        return (featureMap)

    # end

    def main(self, partition):
        featureMap = self.getFeatureMap(partition)
        return (featureMap)

    # end


# end


def getSamplesFeature(probType, mean, sampleNum, stdBias, numThreshold,
                      partitions, totalRow, totalCol):
    # l1 bases
    bases = []
    if probType == "l1c":
        b1 = [1, 1, 1, 1, 1, 1, 0]
        b2 = [1, 1, 1, 1, 0, 0, 0]
        b3 = [1, 1, 0, 0, 0, 0, 0]
        bases = [b1, b2, b3]
    elif probType == "l1":
        b1 = [1, 1, 1, -1, -1, -1, 0]
        b2 = [1, 1, -1, -1, 0, 0, 0]
        b3 = [1, -1, 0, 0, 0, 0, 0]
        bases = [b1, b2, b3]
    elif probType == "lall":
        b1 = [1, 1, 1, 1, 1, 1, 0]
        b2 = [1, 1, 1, 1, 0, 0, 0]
        b3 = [1, 1, 0, 0, 0, 0, 0]
        b4 = [1, 1, 1, -1, -1, -1, 0]
        b5 = [1, 1, -1, -1, 0, 0, 0]
        b6 = [1, -1, 0, 0, 0, 0, 0]
        bases = [b1, b2, b3, b4, b5, b6]
    else:
        base1 = [1, 1, 1, -1, -1, -1, 0]
        base2 = [1, 1, -1, -1, 0, 0, 0]
        base3 = [1, -1, 0, 0, 0, 0, 0]
        base4 = [2, 1, 1, -1, -1, -2, 0]
        bases = [base1, base2, base3, base4]
    baseTypeNum, basesMtrx = getBasesMtrxs(bases)
    randomRowColIdx = getRandomRowColIdx(low=0,
                                         hight=totalCol - 1,
                                         row=2,
                                         col=len(bases[0]),
                                         number=sampleNum)
    # get all base mtrx

    scorePath = "consisBasesScoresL1.npy"
    consisBasesScores = torch.tensor(np.load(scorePath))
    mtx2map = Part23dMap(baseTypeNum, basesMtrx, totalRow, totalCol,
                         randomRowColIdx)
    samples = torch.cat(list(map(mtx2map.getSamples, partitions)))
    labels = torch.tensor(list(map(nonZeroNum, samples)))
    samples = addDataError(samples, mean, stdBias)
    zn, xn, yn = samples.size()
    samples = samples.view(zn, 1, xn, yn)
    labels, samples = addNumMeanNoise(samples, labels, int(len(samples) / 20),
                                      mean, stdBias)
    zn, _, xn, yn = samples.size()
    rowFeatureMap, colFeatureMap = mtx2map.getRowColFeatureMapNoMaxPooling(
        samples.view(zn, xn, yn))
    # combine row and col

    # get opt row and col
    rFM_cBS_nT = [[rfm, consisBasesScores, numThreshold]
                  for rfm in rowFeatureMap]
    cFM_cBS_nT = [[cfm, consisBasesScores, numThreshold]
                  for cfm in colFeatureMap]

    optRowFeatureMap = list(map(getOptFeatureMap, rFM_cBS_nT))
    optColFeatureMap = list(map(getOptFeatureMap, cFM_cBS_nT))
    # complete row and col to the same col number
    optRowFeatureMap = completeData2SameColLen(optRowFeatureMap)
    optColFeatureMap = completeData2SameColLen(optColFeatureMap)
    # combine opt row and col to one
    # reshape optFeatureMap data
    optFeatureMap = torch.stack((optRowFeatureMap, optColFeatureMap), dim=1)
    # reshap featureMap data
    featureMap = torch.stack((rowFeatureMap, colFeatureMap), dim=1)
    return (labels, samples, featureMap, optFeatureMap)


# end


def nonZeroNum(sample):
    if len(torch.nonzero(sample)) >= 6:
        return (len(torch.nonzero(sample)))
    else:
        return (0)


# end


def getLabelsFrmSamples(samples):
    labels = torch.tensor(list(map(nonZeroNum, samples)))
    return (labels)


# end


def delFeature(fTmp, idx, scoreTmp, conThreshold):
    saveIdx = list(np.where(scoreTmp[idx] <= conThreshold))[0]
    fTmp = fTmp[saveIdx]
    scoreTmp = scoreTmp[np.ix_(saveIdx, saveIdx)]
    return ([fTmp, scoreTmp])


# end


def getOptFeatureMap_old(f_b_n):
    st = time.time()
    feature, basesConsisScores, numThreshold = f_b_n[0], f_b_n[1], f_b_n[2]
    feature = torch.from_numpy(feature)
    basesConsisScores = torch.from_numpy(basesConsisScores)
    fSort = feature.T.clone()  # 7*M->M*7
    fSort = torch.sort(fSort)
    fSort = fSort[0]
    # rowNum = fSort.size()[0]
    colNum = fSort.size()[1]
    optFeatures = list()
    for c in range(colNum):
        fMaxMeans = list()
        fTmp = fSort.clone()
        scoreTmp = basesConsisScores.clone()
        while len(fTmp) >= 1:
            fMean = torch.mean(fTmp, dim=1, keepdim=True)
            fMaxMean, fMaxIdx = torch.max(fMean, 0)
            fMaxMeans.append(fMaxMean)
            if len(fMaxMeans) == numThreshold:
                break
            fMaxIdx = fMaxIdx.item()
            fTmp, scoreTmp = delFeature(fTmp, fMaxIdx, scoreTmp, 2)
        optFeatures.append(fMaxMeans)
        if c == colNum:
            break
        fSort = fSort[:, 1:]
    print(time.time() - st)
    print("end")
    # maxLen = max([(len(f)) for f in optFeatures])
    # for f in optFeatures:
    #  f.extend(0.0 for _ in range(maxLen - len(f)))
    return (optFeatures)


# end
# @pysnooper.snoop()
@jit
def getOptFeatureMap(f_b_n):
    feature = f_b_n[0]
    basesConsisScores = f_b_n[1]
    numThreshold = f_b_n[2]
    fSort = feature.T  # 7*M->M*7
    fSortN = list()
    for fs in fSort:
        fSortN.append(sorted(fs))
    fSort = np.array(fSortN)
    # fSort = np.sort(fSort)
    colNum = fSort.shape[1]
    optFeatures = list()
    for c in np.arange(colNum):
        fMaxMeans = []
        fTmp = fSort.copy()
        scoreTmp = basesConsisScores.copy()
        for i in range(1000):
            fMeanTmp = []
            for j in fTmp:
                fMeanTmp.append(j.mean())
            fMean = np.array(fMeanTmp)
            fMaxMean = fMean.max()
            fMaxIdx = fMean.argmax()
            fMaxMeans.append(fMaxMean)
            if len(fMaxMeans) == numThreshold:
                break
            delIdx = []
            for k in range(len(scoreTmp)):
                if scoreTmp[fMaxIdx][k] > 2:
                    delIdx.append(k)
            fTmp = np.delete(fTmp, delIdx)
            scoreTmp = np.delete(scoreTmp, delIdx)
            scoreTmp = scoreTmp.T
            scoreTmp = np.delete(scoreTmp, delIdx)
            # scoreTmp = scoreTmp[np.ix_(saveIdx, saveIdx)]
        optFeatures.append(fMaxMeans)
        if c == colNum:
            break
        fSort = fSort[..., 1:]
    return (optFeatures)


# sparse ssvd
def ssvd(x, r=1):
    from scipy.sparse.linalg import svds
    x = np.array(x)
    sptop = svds(x, k=r)
    sptoptmp = np.zeros((sptop[0].shape[1], sptop[2].shape[0]))
    sptoptmp[np.diag_indices(np.min(sptoptmp.shape))] = sptop[1]
    sptopreconstruct = np.dot(sptop[0], np.dot(sptoptmp, sptop[2]))
    sptopreconstruct = torch.tensor(sptopreconstruct)
    return (sptopreconstruct)


# end

# end


# transtorm 3 1x7 1d bases vectors to 1 7xm 2d matrix
def getBaseMtrx(base):
    from itertools import permutations
    basePermsMtrx = list(set(list(permutations(base, len(base)))))
    basePermsMtrx = np.stack(basePermsMtrx).T
    return (basePermsMtrx)


# end


def getBasesMtrxs(bases):
    from itertools import accumulate
    basesMtrxs = [getBaseMtrx(base) for base in bases]
    baseTypeNum = [basesMtrx.shape[1] for basesMtrx in basesMtrxs]
    basesMtrx = np.hstack(basesMtrxs)  # to 7x(m1+m2+m3)
    baseTypeNum = np.insert(baseTypeNum, 0, 0)
    baseTypeNum = list(accumulate(baseTypeNum))
    return (baseTypeNum, basesMtrx)


# end

def getTimeVector(baseLen):
    norm_samples = list()
    [norm_samples.append(np.random.randn(baseLen)) for _ in range(10000)]
    norm_samples = np.stack(norm_samples)
    norm_samples = abs(norm_samples)
    #print(norm_samples[0:5, :])
    norm_samples = -(np.sort(-norm_samples, axis=1))
    #print(norm_samples[0:5, :])
    norm_vector = np.mean(norm_samples, axis = 0)
    #print(norm_vector)
    norm_vector = 1 / norm_vector
    #print(norm_vector)
    return(norm_vector)

# end

def getBasesMtrxAfterKmean(n_clusters):
    # base1 = [1, -1, 0, 0, 0, 0, 0]
    base2 = [1, 1, -1, -1, 0, 0, 0]
    base3 = [2, 1, 1, -1, -1, -2, 0]
    base4 = [1, 1, 1, -1, -1, -1, 0]
    bases = [base2, base3, base4]
    baseTypeNum, basesMtrx = getBasesMtrxs(bases)
    basesMtrx = pd.DataFrame(basesMtrx)
    basesMtrx = basesMtrx.T
    from sklearn.cluster import KMeans
    kmeansRes = KMeans(n_clusters=n_clusters).fit(basesMtrx)
    # Save the labels
    basesMtrx.loc[:, 'label'] = kmeansRes.labels_
    # multiply dif times to each base type
    basesMtrx.iloc[baseTypeNum[0]:baseTypeNum[1],
                   0:7] = basesMtrx.iloc[baseTypeNum[0]:baseTypeNum[1],
                                         0:7].multiply(1.11)
    basesMtrx.iloc[baseTypeNum[1]:baseTypeNum[2],
                   0:7] = basesMtrx.iloc[baseTypeNum[1]:baseTypeNum[2],
                                         0:7].multiply(0.63)
    # basesMtrx.iloc[baseTypeNum[2]:baseTypeNum[3], 0:7] = basesMtrx.iloc[baseTypeNum[2]:baseTypeNum[3], 0:7].multiply(0.63)

    basesMtrx = basesMtrx.sort_values(by="label", ascending=True)
    baseIdAfterKMeans = basesMtrx.index
    baseTypeNumAfterKmean = [0]
    for i in range(n_clusters):
        typeLen = basesMtrx[basesMtrx["label"] == i].shape[0]
        baseTypeNumAfterKmean.append(baseTypeNumAfterKmean[i] + typeLen)
    return (basesMtrx, baseTypeNumAfterKmean, baseIdAfterKMeans)


# end

def getBasesMtrxAfterKmean_mul(baseLen, n_filter, n_clusters=16):
    bases = list()
    if baseLen == 5:
        bases.append([1,-1,0,0,0])
        bases.append([1,1,-1,-1,0])
    if baseLen == 7:
        bases.append(np.array([1,1,-1,-1,0,0,0])*1)
        bases.append(np.array([1,1,1,-1,-1,-1,0])*(0.86))
        bases.append(np.array([-2,-1,0,0,0,1,2])*(0.61))
        bases.append(np.array([-2,-1,-1,0,1,1,2])*(0.55))
        bases.append(np.array([-5,-1,0,0,0,1,5])*(0.28))
        bases.append(np.array([-5,-2,-1,0,1,2,5])*(0.24))
    if baseLen == 9:
        bases.append(np.array([1,1,-1,-1,0,0,0,0,0])*1)
        bases.append(np.array([1,1,1,-1,-1,-1,0,0,0])*0.81)
        bases.append(np.array([1,1,1,1,-1,-1,-1,-1,0])*0.74)
        bases.append(np.array([-2,-1,1,2,0,0,0,0,0])*0.62)
        bases.append(np.array([-2,-1,-1,1,1,2,0,0,0])*0.54)
        bases.append(np.array([-2,-2,-1,-1,1,1,2,2,0])*0.31)
    if baseLen == 11:
        bases.append([1,1,-1,-1,0,0,0,0,0,0,0])
        bases.append([1,1,1,-1,-1,-1,0,0,0,0,0])
        bases.append([1,1,1,1,-1,-1,-1,-1,0,0,0])
        bases.append([1,1,1,1,1,-1,-1,-1,-1,-1,0])
    if baseLen == 13:
        bases.append([1,1,1,-1,-1,-1,0,0,0,0,0,0,0])
        bases.append([1,1,1,1,-1,-1,-1,-1,0,0,0,0,0])
        bases.append([1,1,1,1,1,-1,-1,-1,-1,-1,0,0,0])
        bases.append([1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,0])
    bases = bases[0:n_filter]
    basesTypeNum, basesMtrx = getBasesMtrxs(bases)
    basesMtrx = pd.DataFrame(basesMtrx)
    basesMtrx = basesMtrx.T
    # kmeans on bases
    from sklearn.cluster import KMeans
    kmeansRes = KMeans(n_clusters=16).fit(basesMtrx)

    # Save the labels
    basesMtrx.loc[:, 'label'] = kmeansRes.labels_

    # multiply dif times to each base type
    #timeVector = getTimeVector(baseLen)
    #basesMtrx.iloc[:, 0:baseLen] = basesMtrx.iloc[:,0:baseLen] * timeVector

    # sort the value by label
    basesMtrx = basesMtrx.sort_values(by="label", ascending=True)
    baseIdAfterKMeans = basesMtrx.index
    baseTypeNumAfterKmean = [0]
    for i in range(n_clusters):
        typeLen = basesMtrx[basesMtrx["label"] == i].shape[0]
        baseTypeNumAfterKmean.append(baseTypeNumAfterKmean[i] + typeLen)
    return (basesMtrx, baseTypeNumAfterKmean, baseIdAfterKMeans)
#end



def getResortMeanFeatureMap(featureMap):
    waitColMean = list()
    xn = featureMap.shape[2]
    for i in range(1, xn + 1):
        currentColMean = np.mean(featureMap[:, :, 0:i, :], axis=2)
        zn, xn, yn = currentColMean.shape
        currentColMean = np.reshape(currentColMean, (zn, xn, 1, yn))
        waitColMean.append(currentColMean)
    featureMap = np.dstack(waitColMean)
    return (featureMap)


# end


def myMaxPooling(featureMap, baseTypeNumAfterKmean):
    featureMap_tmp = list()
    for i in range(len(baseTypeNumAfterKmean) - 1):
        maxSlice = np.max(
            featureMap[:, :, :,
                       baseTypeNumAfterKmean[i]:baseTypeNumAfterKmean[i + 1]],
            axis=-1)
        zn, xn, yn = maxSlice.shape
        maxSlice = np.reshape(maxSlice, [zn, xn, yn, 1])
        featureMap_tmp.append(maxSlice)
    featureMap_tmp = np.concatenate(featureMap_tmp, axis=-1)
    return (featureMap_tmp)


# end


def getLabelFrSamples(samples, xn, yn):
    labels = list()
    xn_sample, yn_sample = samples[0][0].shape
    for s1 in samples:
        for s2 in s1:
            n1 = np.sum(s2.index < xn)
            n2 = np.sum(s2.columns < yn)
            lab = n1 * n2
            '''
            if (lab < 25):
                labels.append(0)
            else:
                labels.append(1)
            '''
            if (lab == 0):
                labels.append(0)
            elif (lab < 25):
                labels.append(1)
            else:
                labels.append(2)


            '''
            if(((n1*n2)/(xn_sample*yn_sample))<0.6):
                labels.append(0)
            else:
                labels.append(1)
            '''
    return (labels)


# end


def getLabelFrSamples_mPattern(samples, xn, yn, totalRow):
    n_block = int(totalRow / xn)
    labelRange = [0, xn]
    for i in range(2, n_block + 1):
        labelRange.append(i * xn)
    labels = list()
    #xn_sample, yn_sample = samples[0][0].shape
    for s1 in samples:
        for s2 in s1:
            getLabel = 0
            for i in range(n_block):
                n1 = np.sum((labelRange[i] < s2.index)
                            & (s2.index < labelRange[i + 1]))
                n2 = np.sum((0 <= s2.columns) & (s2.columns < yn))
                lab = n1 * n2
                if (lab >= 25):
                    labels.append(i + 1)
                    getLabel = 1
                    break
            if (getLabel == 0):
                labels.append(0)
    return (labels)


# end


def getNewPart(samples, mateData, xn):
    # get all row index
    #label, samples = samples
    allRowIdx = list()
    allColIdx = list()
    for sample in samples:
        if len(sample) == 0:
            return ([])
        allRowIdx.append(list(sample.index))
        allColIdx.append(list(sample.columns))
    # count row index frequency
    allRowIdx = np.concatenate(allRowIdx)
    allColIdx = np.concatenate(allColIdx)
    allRowIdx = np.unique(allRowIdx)
    allColIdx = np.unique(allColIdx)
    freqMtrx = np.zeros((len(allRowIdx), len(allColIdx)))
    freqMtrx = pd.DataFrame(freqMtrx, index=allRowIdx, columns=allColIdx)
    for sample in samples:
        for i in sample.index:
            for j in sample.columns:
                if i == j:
                    continue
                freqMtrx.loc[i, j] = freqMtrx.loc[i, j] + 1
    if (xn > np.min(freqMtrx.shape)):
        xn = np.min(freqMtrx.shape)
    value = np.argpartition(freqMtrx.values.ravel(), -xn)[-xn:]
    i2d = np.unravel_index(value, freqMtrx.shape)
    maxRowCountIdx = i2d[0]
    maxColCountIdx = i2d[1]
    # get most common row index
    '''
    from collections import Counter
    maxRowCount = Counter(allRowIdx).most_common(xn)
    maxRowCountIdx = [mc[0] for mc in maxRowCount]

    maxColCount = Counter(allColIdx).most_common(xn)
    maxColCountIdx = [mc[0] for mc in maxColCount]
    '''
    newPart = mateData.loc[maxRowCountIdx, maxColCountIdx]
    '''
    # get the columns from the common row index
    allColIdx = list()
    for sample in samples:
        # the length of maxCountIndex and columns of sample
        interLen = len(list(set(maxRowCountIdx).intersection(set(list(sample.index)))))
        if interLen > 0:
            allColIdx.append(list(sample.columns))
    allColIdx = list(set(np.concatenate(allColIdx)))

    # get new partition
    newPart = mateData.loc[maxRowCountIdx, allColIdx]
    # update partition by theshold svd
    waitColIdx = list(set(list(mateData.columns)) - set(allColIdx))
    for colIdx in waitColIdx:

        if len(newPart.columns) >= xn:
            break

        _, s, _ = np.linalg.svd(newPart.values)
        waitCol = mateData.loc[maxRowCountIdx, colIdx].values.reshape(len(maxRowCountIdx), 1)
        tmpPart = np.concatenate((newPart.values, waitCol), axis=1)
        _, s_new, _ = np.linalg.svd(tmpPart)
        #if s_new[1]/s_new[0] <= s[1]/s[0]:
        if s_new[0] >= s[0] and s_new[1] <= s[1]:
            newPart.loc[maxRowCountIdx, colIdx] = mateData.loc[maxRowCountIdx, colIdx]

    allColIdx = list(newPart.columns)

    # update the new partition thru row side
    waitRowIdx = list(set(list(mateData.index)) - set(maxRowCountIdx))
    for rowIdx in waitRowIdx:

        if len(newPart.index) >= xn:
            break

        _, s, _ = np.linalg.svd(newPart.values)
        waitRow = mateData.loc[rowIdx, allColIdx].values.reshape(1, len(allColIdx))
        tmpPart = np.concatenate((newPart.values, waitRow), axis=0)
        _, s_new, _ = np.linalg.svd(tmpPart)
        #if s_new[1]/s_new[0] <= s[1]/s[0]:
        if s_new[0] >= s[0] and s_new[1] <= s[1]:
            newPart.loc[maxRowCountIdx, colIdx] = mateData.loc[maxRowCountIdx, colIdx]
            newPart.loc[rowIdx, allColIdx] = mateData.loc[rowIdx, allColIdx]
    '''
    return (newPart)


# end

def getNewPart_clustering(samples, mateData, n_cluster):
    # get all row index
    samples = samples
    if len(samples) > 500:
        n_cluster = n_cluster * 2
    allRowIdx = list()
    allColIdx = list()
    for sample in samples:
        if len(sample) == 0:
            return ([])
        allRowIdx.append(list(sample.index))
        allColIdx.append(list(sample.columns))
    # count row index frequency
    allRowIdx = np.concatenate(allRowIdx)
    allColIdx = np.concatenate(allColIdx)
    allRowIdx = np.unique(allRowIdx)
    allColIdx = np.unique(allColIdx)
    freqMtrx = np.zeros((len(allRowIdx), len(allColIdx)))
    freqMtrx = pd.DataFrame(freqMtrx, index=allRowIdx, columns=allColIdx)
    for sample in samples:
        for i in sample.index:
            for j in sample.columns:
                if i == j:
                    continue
                freqMtrx.loc[i, j] = freqMtrx.loc[i, j] + 1

    #freqData = (freqMtrx / np.max(freqMtrx)).copy()
    from sklearn.cluster import SpectralBiclustering
    SpectralBicluster = SpectralBiclustering(n_clusters=(n_cluster,n_cluster)).fit(freqMtrx)
    
    rowIdxSet = [[] for i in range(n_cluster)]
    colIdxSet = [[] for i in range(n_cluster)]
    for i in range(len(allRowIdx)):
        lb = SpectralBicluster.row_labels_[i]
        rowIdxSet[lb].append(allRowIdx[i])

    for  i in range(len(allColIdx)):
        lb = SpectralBicluster.column_labels_[i]
        colIdxSet[lb].append(allColIdx[i])
    newParts = list()

    newParts = list(map(lambda rowColIdx:mateData.loc[rowColIdx[0],rowColIdx[1]], zip(rowIdxSet,colIdxSet) ))
    '''
    for rowIdxs, colIdxs in zip(rowIdxSet,colIdxSet):
        newParts.append(mateData.loc[rowIdxs,colIdxs])
    '''
    '''
    for rowIdxs in rowIdxSet:
        for colIdxs in colIdxSet:
            newParts.append(mateData.loc[rowIdxs, colIdxs])
    '''
    #newPart = mateData.loc[maxRowCountIdx, maxColCountIdx]
    return (newParts)


# end





def getConsistencyScoreMtrx(basesMtrx):
    conBasesMtrx = basesMtrx.t().clone()
    rowNum = conBasesMtrx.size()[0]
    colNum = conBasesMtrx.size()[1]
    consisScoreMtrx = torch.diag(torch.tensor([7] * rowNum), diagonal=0)
    for i1 in range(rowNum - 1):
        for i2 in range(i1 + 1, rowNum):
            score = 0
            for j in range(colNum):
                if conBasesMtrx[i1][j] == conBasesMtrx[i2][j]:
                    score = score + 1
                else:
                    score = score - 1
            consisScoreMtrx[i1][i2] = consisScoreMtrx[i2][i1] = score
    return (consisScoreMtrx)


# end


# get all row index list and col index list randomly
@lru_cache()
def getRandomRowColIdx(low=0, high=49, row=2, col=7, number=10):
    randomRowIdx = [
        np.sort(np.random.choice(range(low, high + 1), col, replace=False))
        for _ in range(number)
    ]
    randomColIdx = [
        np.sort(np.random.choice(range(low, high + 1), col, replace=False))
        for _ in range(number)
    ]
    randomRowColIdx = list(zip(randomRowIdx, randomColIdx))
    return (randomRowColIdx)


# end

# get all row index list and col index list randomly


# merge 2 3d matries
def merge2Mtrx(smallMtrx, bigMtrx, r, c, replace=0):
    # the size of a and b can be different
    z = 0
    if replace == 0:
        bigMtrx[z:smallMtrx.shape[0], r:r + smallMtrx.shape[1],
                c:c + smallMtrx.shape[2]] += smallMtrx
    else:
        bigMtrx[z:smallMtrx.shape[0], r:r + smallMtrx.shape[1],
                c:c + smallMtrx.shape[2]] = smallMtrx
    return (bigMtrx)


# get l1c mean blocks data background is zero
def getL1CMeanData(mean,
                   stdBias,
                   noiseNorm,
                   noiseMbias,
                   noiseStdbias,
                   num,
                   zn,
                   xn,
                   yn,
                   totalRow,
                   totalCol,
                   overlap=0):
    std = torch.ones(zn, xn, yn).add_(stdBias)
    # noise parameters
    noiseMean = mean + noiseMbias
    noisestd = torch.ones(zn, totalRow, totalCol).add_(noiseStdbias)
    noiseData = torch.normal(noiseMean, noisestd)

    # prepare data
    labels_datas = list()
    for i in range(1, num + 1):
        label = i
        blocks = list()
        blocks = [
            (torch.normal(mean, std) + (torch.randn(zn, xn, yn) * noiseNorm))
            for k in range(i)
        ]

        addNoiseRes = merge2Mtrx(blocks[0], noiseData, 0, 0)
        if i == 1:
            labels_datas.append([label, addNoiseRes.clone()])
            continue

        if overlap == 0:
            r, c = xn, yn
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c)

                r, c = r + xn, c + yn
            labels_datas.append([label, addNoiseRes.clone()])
        else:
            r, c = int(xn / 2) + 1, int(yn / 2) + 1
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c)
                r, c = r + int(xn / 2) + 1, c + int(yn / 2) + 1
            labels_datas.append([label, addNoiseRes.clone()])
    return (labels_datas)


# get l1c normal blocks data
def getL1CNormData(normalBias, minusMean, num, zn, xn, yn, totalRow, totalCol,
                   overlap, replace):
    # noise parameters
    gaussianNoise = torch.randn(zn, totalRow, totalCol)
    gaussianNoise = gaussianNoise - torch.mean(gaussianNoise)
    # gaussianNoise = torch.zeros(zn, totalRow, totalCol)  # zero noise
    # prepare normal distribution data
    labels_datas = list()
    for i in range(1, num + 1):
        label = i
        blocks = list()
        if minusMean == 0:
            blocks = [torch.randn(zn, xn, yn) + normalBias for _ in range(i)]
        else:
            for _ in range(i):
                block = torch.randn(zn, xn, yn)
                block = block - torch.mean(block) + normalBias
                blocks.append(block)

        addNoiseRes = merge2Mtrx(blocks[0], gaussianNoise, 0, 0, replace)
        if i == 1:
            labels_datas.append([label, addNoiseRes.clone()])
            continue
        if overlap == 0:
            r, c = xn, yn
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, replace)
                r, c = r + xn, c + yn
            labels_datas.append([label, addNoiseRes.clone()])
        else:
            r, c = int(xn / 2) + 1, int(yn / 2) + 1
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, 1)
                r, c = r + int(xn / 2) + 1, c + int(yn / 2) + 1
            labels_datas.append([label, addNoiseRes.clone()])
    return (labels_datas)


#  get l1 mean blocks data background is 0
def getL1MeanData(mean,
                  stdBias,
                  noiseNorm,
                  noiseMbias,
                  noiseStdbias,
                  num,
                  zn,
                  xn,
                  yn,
                  totalRow,
                  totalCol,
                  overlap=0):
    std = torch.ones(1, yn).add_(stdBias)
    blocksNum = num * zn
    blocks = [((torch.normal(mean, std).t() * torch.normal(mean, std)) +
               (torch.randn(xn, yn) * noiseNorm)) for _ in range(blocksNum)]
    blocks = torch.stack(blocks).view(num, zn, xn, yn)

    # noise parameters
    noiseMean = int(torch.mean(blocks) + noiseMbias)
    noisestd = torch.ones(zn, totalRow, totalCol).add_(noiseStdbias)
    noiseData = torch.normal(noiseMean, noisestd)
    # noiseData = torch.zeros(zn, totalRow, totalCol)
    labels_datas = list()
    for i in range(1, num + 1):
        label = i
        addNoiseRes = merge2Mtrx(blocks[0], noiseData, 0, 0)
        if i == 1:
            labels_datas.append([label, addNoiseRes.clone()])
            continue
        if overlap == 0:
            r, c = xn, yn
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c)
                r, c = r + xn, c + yn
            labels_datas.append([label, addNoiseRes.clone()])

        else:
            r, c = int(xn / 2) + 1, int(yn / 2) + 1
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c)
                r, c = r + int(xn / 2) + 1, c + int(yn / 2) + 1
            labels_datas.append([label, addNoiseRes.clone()])

    return (noiseMean, labels_datas)


#  get lk normal blocks data background noise is gaussian noise
def getLkNormData(lk, normalBias, minusMean, num, zn, xn, yn, totalRow,
                  totalCol, overlap, replace):
    blocksNum = num * zn
    blocks = list()
    if minusMean == 0:
        blocks = [
            torch.matmul(torch.randn(xn, lk), torch.randn(lk, xn)) + normalBias
            for _ in range(blocksNum)
        ]
    else:
        for _ in range(blocksNum):
            block = torch.matmul(torch.randn(xn, lk), torch.randn(lk, xn))
            block = block - torch.mean(block) + normalBias
            blocks.append(block)
    blocks = torch.stack(blocks).view(num, zn, xn, yn)
    # gaussian noise parameters
    gaussianNoise = torch.randn(zn, totalRow, totalCol)
    gaussianNoise = gaussianNoise - torch.mean(gaussianNoise)
    # gaussianNoise = torch.zeros(zn, totalRow, totalCol)  # zero background
    # noise

    labels_datas = list()
    for i in range(1, num + 1):
        label = i
        addNoiseRes = merge2Mtrx(blocks[0], gaussianNoise, 0, 0, replace)
        if i == 1:
            labels_datas.append([label, addNoiseRes.clone()])
            continue
        if overlap == 0:
            r, c = xn, yn
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, replace)
                r, c = r + xn, c + yn
            labels_datas.append([label, addNoiseRes.clone()])

        else:
            r, c = int(xn / 2) + 1, int(yn / 2) + 1
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, 1)
                r, c = r + int(xn / 2) + 1, c + int(yn / 2) + 1
            labels_datas.append([label, addNoiseRes.clone()])
    return (labels_datas)


#  get l1 base blocks data background noise is gaussian noise
def getL1SpeBaseData(baseLen, crType, minusMean, errorStdBias, blockNum, baseTimes, zn,
                     xn, yn, totalRow, totalCol, overlap, replace):
    #torch.manual_seed(16)
    blocksNum = blockNum * zn
    mean = 0
    #errorStd = torch.zeros(xn, yn).add_(errorStdBias)
    #error = torch.normal(mean, errorStd)

    error = torch.tensor(np.random.normal(loc=0.0, scale=errorStdBias, size=(xn,yn)))
    blocks = list()
    variance = list()
    bases = list()
    if crType == "uniform":
        if minusMean == 1:
            for _ in range(blocksNum):
                c = torch.rand(xn, 1)
                r = torch.rand(1, xn)
                block = torch.matmul(c, r)
                block = block - torch.mean(block)
                block = block + torch.ones_like(block)*baseTimes + error
                variance.append(torch.var(block).numpy())
                blocks.append((block).numpy())
        else:
            for _ in range(blocksNum):
                c = torch.rand(xn, 1)
                r = torch.rand(1, xn)
                block = torch.matmul(c, r) * baseTimes
                variance.append(torch.var(block).numpy())
                blocks.append((block + error).numpy())
    elif crType == "norm":
        if minusMean == 1:
            for _ in range(blocksNum):
                c = torch.randn(xn, 1)
                r = torch.randn(1, xn)
                block = torch.matmul(c, r) * baseTimes
                block = block - torch.mean(block)
                variance.append(torch.var(block).numpy())
                blocks.append((block + error).numpy())
        else:
            for _ in range(blocksNum):
                c = torch.randn(xn, 1)
                r = torch.randn(1, xn)
                block = torch.matmul(c, r) * baseTimes
                variance.append(torch.var(block).numpy())
                blocks.append((block + error).numpy())
    else:
        base1 = [1] * int(xn * (2 / 5))
        base2 = [-1] * int(xn * (2 / 5))
        base3 = [0] * int(xn * (1 / 5))
        bases = np.concatenate((base1, base2, base3))
        bases = torch.tensor(bases)
        if minusMean == 1:
            for _ in range(blocksNum):
                rdmRowIdx = np.random.choice(range(xn), xn, replace=False)
                rdmColIdx = np.random.choice(range(xn), xn, replace=False)
                c = bases[rdmColIdx].view(xn, 1).float()
                r = bases[rdmRowIdx].view(1, xn).float()
                block = torch.matmul(c, r) * baseTimes
                block = block - torch.mean(block)
                variance.append(torch.var(block).numpy())
                blocks.append((block + error).numpy())
        else:
            for _ in range(blocksNum):
                rdmRowIdx = np.random.choice(range(xn), xn, replace=False)
                rdmColIdx = np.random.choice(range(xn), xn, replace=False)
                c = bases[rdmColIdx].view(xn, 1).float()
                r = bases[rdmRowIdx].view(1, xn).float()
                block = torch.matmul(c, r) * baseTimes + error
                variance.append(torch.var(block).numpy())
                blocks.append(block.numpy())
    blocks = np.stack(blocks)
    blocks = np.reshape(blocks, (blockNum, zn, xn, yn))
    # noise parameters
    c=np.random.normal(loc=0.0, scale=1.0, size=(xn,1))
    r=np.random.normal(loc=0.0, scale=1.0, size=(1,yn))
    block = np.matmul(c,r)
    block_std = np.std(block)
    bgNoise = np.random.normal(loc=0.0, scale=block_std, size=(zn, totalRow, totalCol))
    bgNoise = bgNoise - np.mean(bgNoise)
    var_noise = np.var(bgNoise)
    '''
    gaussianNoise = np.random.randn(zn, totalRow, totalCol)
    gaussianNoise = gaussianNoise - np.mean(gaussianNoise)
    var_noise = np.var(gaussianNoise)
    zeroNoise = np.zeros((zn, totalRow, totalCol))  # zero background noise
    # gaussianNoise = zeroNoise
    '''
    labels_datas = list()
    for i in range(1, blockNum + 1):
        label = i
        addNoiseRes = merge2Mtrx(blocks[0], bgNoise, 0, 0, replace)
        if i == 1:
            labels_datas.append([label, addNoiseRes.copy()])
            continue
        if overlap == 0:
            r, c = xn, yn
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, replace)
                r, c = r + xn, c + yn
            labels_datas.append([label, addNoiseRes.copy()])
        else:
            r, c = int(xn / 2) + 1, int(yn / 2) + 1
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, 1)
                r, c = r + int(xn / 2) + 1, c + int(yn / 2) + 1
            labels_datas.append([label, addNoiseRes.copy()])
    return (labels_datas, variance, var_noise)


#  get lK base blocks data background noise is gaussian noise
def getLKSpeBaseData(crType,
                     minusMean,
                     errorStdBias,
                     blockNum,
                     baseTimes,
                     zn,
                     xn,
                     yn,
                     totalRow,
                     totalCol,
                     overlap,
                     replace,
                     K=2):
    np.random.seed(16)
    torch.manual_seed(16)
    blocksNum = blockNum * zn
    mean = 0
    errorStd = torch.zeros(xn, yn).add_(errorStdBias)
    error = torch.normal(mean, errorStd)
    blocks = list()
    variance = list()
    bases = list()
    if crType == "uniform":
        if minusMean == 1:
            for _ in range(blocksNum):
                c = torch.rand(xn, K)
                r = torch.rand(K, xn)
                block = torch.matmul(c, r) * baseTimes
                block = block - torch.mean(block)
                variance.append(torch.var(block).numpy())
                blocks.append((block + error).numpy())
        else:
            for _ in range(blocksNum):
                c = torch.rand(xn, K)
                r = torch.rand(K, xn)
                block = torch.matmul(c, r) * baseTimes
                variance.append(torch.var(block).numpy())
                blocks.append((block + error).numpy())
    elif crType == "norm":
        if minusMean == 1:
            for _ in range(blocksNum):
                c = torch.randn(xn, K)
                r = torch.randn(K, xn)
                block = torch.matmul(c, r) * baseTimes
                block = block - torch.mean(block)
                variance.append(torch.var(block).numpy())
                blocks.append((block + error).numpy())
        else:
            for _ in range(blocksNum):
                c = torch.randn(xn, K)
                r = torch.randn(K, xn)
                block = torch.matmul(c, r) * baseTimes
                variance.append(torch.var(block).numpy())
                blocks.append((block + error).numpy())
    else:
        base1 = [1] * int(xn * (2 / 5))
        base2 = [-1] * int(xn * (2 / 5))
        base3 = [0] * int(xn * (1 / 5))
        bases = np.concatenate((base1, base2, base3))
        bases = torch.tensor(bases)
        if minusMean == 1:
            for _ in range(blocksNum):
                rdmRowIdx = np.random.choice(range(xn * K),
                                             xn * K,
                                             replace=False)
                rdmColIdx = np.random.choice(range(xn * K),
                                             xn * K,
                                             replace=False)
                c = bases[rdmColIdx].view(xn, K).float()
                r = bases[rdmRowIdx].view(K, xn).float()
                block = torch.matmul(c, r) * baseTimes
                block = block - torch.mean(block)
                variance.append(torch.var(block).numpy())
                blocks.append((block + error).numpy())
        else:
            for _ in range(blocksNum):
                rdmRowIdx = np.random.choice(range(xn * K),
                                             xn * K,
                                             replace=False)
                rdmColIdx = np.random.choice(range(xn * K),
                                             xn * K,
                                             replace=False)
                c = bases[rdmColIdx].view(xn, 1).float()
                r = bases[rdmRowIdx].view(1, xn).float()
                block = torch.matmul(c, r) * baseTimes + error
                variance.append(torch.var(block).numpy())
                blocks.append(block.numpy())
    # u,s,v = np.linalg.svd(blocks[0])
    # print(s)
    # sys.exit()
    blocks = np.stack(blocks)
    blocks = np.reshape(blocks, (blockNum, zn, xn, yn))
    # noise parameters
    gaussianNoise = np.random.randn(zn, totalRow, totalCol)
    gaussianNoise = gaussianNoise - np.mean(gaussianNoise)
    var_noise = np.var(gaussianNoise)
    zeroNoise = np.zeros((zn, totalRow, totalCol))  # zero background noise
    # gaussianNoise = zeroNoise
    labels_datas = list()
    for i in range(1, blockNum + 1):
        label = i
        addNoiseRes = merge2Mtrx(blocks[0], gaussianNoise, 0, 0, replace)
        if i == 1:
            labels_datas.append([label, addNoiseRes.copy()])
            continue
        if overlap == 0:
            r, c = xn, yn
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, replace)
                r, c = r + xn, c + yn
            labels_datas.append([label, addNoiseRes.copy()])
        else:
            r, c = int(xn / 2) + 1, int(yn / 2) + 1
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, 1)
                r, c = r + int(xn / 2) + 1, c + int(yn / 2) + 1
            labels_datas.append([label, addNoiseRes.copy()])
    return (labels_datas, variance, var_noise)


# end


#  get Lk and M patterns data background noise is gaussian noise
def getLKAndMPatternData(crType,
                         minusMean,
                         errorStdBias,
                         blockNum,
                         baseTimes,
                         zn,
                         xn,
                         yn,
                         totalRow,
                         totalCol,
                         overlap,
                         replace,
                         K=1):
    blocksNum = blockNum * zn
    mean = 0
    errorStd = torch.zeros(xn, yn).add_(errorStdBias)
    error = torch.normal(mean, errorStd)
    blocks = list()
    variance = list()
    bases = list()
    if crType == "uniform":
        if minusMean == 1:
            c = torch.rand(xn, K)
            r = torch.rand(K, xn)
            for m in range(0, blocksNum):
                block = torch.matmul(c, r)
                block = block - torch.mean(block)
                #variance.append(torch.var(block).numpy())
                #blocks.append((block + (error * m) ).numpy())
                if m == 0:
                    blocks.append(block.numpy())
                else:
                    blocks.append((block + error + baseTimes).numpy())

        else:
            for _ in range(blocksNum):
                c = torch.rand(xn, K)
                r = torch.rand(K, xn)
                block = torch.matmul(c, r) * baseTimes
                variance.append(torch.var(block).numpy())
                blocks.append((block + error).numpy())
    elif crType == "norm":
        if minusMean == 1:
            for _ in range(blocksNum):
                c = torch.randn(xn, K)
                r = torch.randn(K, xn)
                block = torch.matmul(c, r) * baseTimes
                block = block - torch.mean(block)
                variance.append(torch.var(block).numpy())
                blocks.append((block + error).numpy())
        else:
            for _ in range(blocksNum):
                c = torch.randn(xn, K)
                r = torch.randn(K, xn)
                block = torch.matmul(c, r) * baseTimes
                variance.append(torch.var(block).numpy())
                blocks.append((block + error).numpy())
    else:
        base1 = [1] * int(xn * (2 / 5))
        base2 = [-1] * int(xn * (2 / 5))
        base3 = [0] * int(xn * (1 / 5))
        bases = np.concatenate((base1, base2, base3))
        bases = torch.tensor(bases)
        if minusMean == 1:
            for _ in range(blocksNum):
                rdmRowIdx = np.random.choice(range(xn * K),
                                             xn * K,
                                             replace=False)
                rdmColIdx = np.random.choice(range(xn * K),
                                             xn * K,
                                             replace=False)
                c = bases[rdmColIdx].view(xn, K).float()
                r = bases[rdmRowIdx].view(K, xn).float()
                block = torch.matmul(c, r) * baseTimes
                block = block - torch.mean(block)
                variance.append(torch.var(block).numpy())
                blocks.append((block + error).numpy())
        else:
            for _ in range(blocksNum):
                rdmRowIdx = np.random.choice(range(xn * K),
                                             xn * K,
                                             replace=False)
                rdmColIdx = np.random.choice(range(xn * K),
                                             xn * K,
                                             replace=False)
                c = bases[rdmColIdx].view(xn, 1).float()
                r = bases[rdmRowIdx].view(1, xn).float()
                block = torch.matmul(c, r) * baseTimes + error
                variance.append(torch.var(block).numpy())
                blocks.append(block.numpy())
    # u,s,v = np.linalg.svd(blocks[0])
    # print(s)
    # sys.exit()
    blocks = np.stack(blocks)
    blocks = np.reshape(blocks, (blockNum, zn, xn, yn))
    # noise parameters
    '''
    gaussianNoise = np.random.randn(zn, totalRow, totalCol)
    gaussianNoise = gaussianNoise - np.mean(gaussianNoise)
    var_noise = np.var(gaussianNoise)
    zeroNoise = np.zeros((zn, totalRow, totalCol))  # zero background noise
    #gaussianNoise = zeroNoise
    '''
    c = np.random.rand(xn,K)
    r = np.random.rand(K, xn)
    std = np.std(np.matmul(c,r))
    bgNoise = np.random.normal(loc=0.0, scale=std, size=(zn, totalRow, totalCol))
    var_noise = np.var(bgNoise)
    labels_datas = list()
    for i in range(1, blockNum + 1):
        label = i
        addNoiseRes = merge2Mtrx(blocks[0], bgNoise, 0, 0, replace)
        if i == 1:
            labels_datas.append([label, addNoiseRes.copy()])
            continue
        if overlap == 0:
            r, c = xn, 0
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, replace)
                r, c = r + xn, 0
            labels_datas.append([label, addNoiseRes.copy()])
        else:
            r, c = int(xn / 2) + 1, int(yn / 2) + 1
            for j in range(1, i):
                addNoiseRes = merge2Mtrx(blocks[j], addNoiseRes, r, c, 1)
                r, c = r + int(xn / 2) + 1, c + int(yn / 2) + 1
            labels_datas.append([label, addNoiseRes.copy()])
    return (labels_datas, variance, var_noise)


# add many mean noise to the whole data
def addNumMeanNoise(data, label, num, mean, stdBias):
    std = torch.zeros(num,
                      data.size()[1],
                      data.size()[2],
                      data.size()[3]).add_(stdBias)
    noise = torch.normal(mean, std)
    noiseLabel = torch.tensor([
        0.0,
    ] * num).long()
    data = torch.cat((data, noise), 0)
    label = torch.cat((label, noiseLabel), 0)
    return (label, data)


# add many mean noise to the whole data
def addNumGaussianNoise(data, label, num):
    gaussianNoise = torch.randn_like(data)
    gaussianNoise = gaussianNoise[0:num]
    gaussianNoise = gaussianNoise - torch.mean(gaussianNoise)
    noiseLabel = torch.tensor([
        0.0,
    ] * num).long()
    data = torch.cat((data, gaussianNoise), 0)
    label = torch.cat((label, noiseLabel), 0)
    return (label, data)


# combine labels and data
def combineLabelData(labels_datas, zn, num):
    labels = torch.cat(
        [torch.tensor([labels_datas[i][0]] * zn).long() for i in range(num)])
    datas = torch.cat([labels_datas[i][1] for i in range(num)])
    return (labels, datas)


# shuffle data
def shuffleData(inputData):
    rowNum = inputData.shape[0]
    colNum = inputData.shape[1]
    rowNum = torch.randperm(rowNum)
    colNum = torch.randperm(colNum)
    inputData = inputData[rowNum[:, None], colNum]
    idx = torch.randperm(inputData.nelement())
    return (inputData.view(-1)[idx].view(inputData.size()))


# separate data into train and test
def separateData(labels, datas, sep):
    trainData = list()
    trainLabel = list()
    testData = list()
    testLabel = list()

    dataLen = datas.size()[0]
    for i in range(dataLen):
        if i % sep == 0:
            testData.append(datas[i])
            testLabel.append(labels[i])
        else:
            trainData.append(datas[i])
            trainLabel.append(labels[i])
    trainData = torch.stack(trainData)
    trainLabel = torch.stack(trainLabel)
    testData = torch.stack(testData)
    testLabel = torch.stack(testLabel)
    return (trainData, trainLabel, testData, testLabel)


def getCNNOutSize(inDim, layerNum, kernelSize):
    for i in range(layerNum - 1):
        inDim = int((inDim - (kernelSize - 1)) / 2)
    return (inDim)


def addDataError(data, mean, stdBias):
    std = torch.zeros_like(data).add_(stdBias)
    noise = torch.normal(mean, std)
    noise = noise - torch.mean(noise)
    return (data + noise)


# end


def completeData2SameColLen(data):
    maxColLen = 0
    for d in data:
        tmpLen = len(d[0])
        maxColLen = tmpLen if tmpLen > maxColLen else maxColLen
    for d in data:
        for col in d:
            col.extend(0.0 for _ in range(maxColLen - len(col)))
    data = torch.tensor(data)
    return (data)


# end
