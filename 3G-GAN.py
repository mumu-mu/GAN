# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 09:50:46 2019

@author: lenovo
"""
import random
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import torchvision.models as models

#load the data
dataF = 'C://Users//lenovo//Desktop//毕设//资料//Salinas_corrected.mat'
dataRow = sio.loadmat(dataF)["salinas_corrected"]
dataTruthF = 'C://Users//lenovo//Desktop//毕设//资料//Salinas_gt.mat'
dataTruth = sio.loadmat(dataTruthF)["salinas_gt"]

#=============PCA=============#
def pca(dataMat,topNfeat=9999999):
    #dataMat：input
    #topNfeat：output top N features
    meanVals = np.mean(dataMat, axis=0)  #to compute the mean value
    meanRemoved = dataMat - meanVals  #to remove the mean value
    covMat = np.cov(meanRemoved, rowvar=0)   #to compute the covariance matrix
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))  #to compute the eigenvalues and eigenvectors
    eigValInd = np.argsort(eigVals)   #to sort from the smallest to the largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #to get the nth largest eigenvalues and eigenvectors
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved * redEigVects   #to transform into low dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    cumCont = sum(eigVals[eigValInd])/sum(eigVals)   #to compute cumulative contribution
    return lowDDataMat,reconMat,cumCont

# to draw principal component
imData = dataRow.reshape([111104, 204])  #reshape the data into 2_D
lowDataMat,reconMat,cumCont = pca(imData, 3)
prinCom = np.zeros([3, 512, 217])
for i in range(3):
    prinCom[i,:,:] = lowDataMat[:,i].reshape([512, 217]) 
[nBand,nRow, nColumn] = prinCom.shape

#to flip and expand the size of data
def flip(data):
    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data

nTrain = 200   #the size of training data 
nTest = 500   #the size of testing data
traEpo = 500   #training epoches
batSize = 200   #batch size
nz = 100   #the band number of noise
ngf = 64   
nc = 3   #the bands number of the output of genetator
numLabel = int(np.max(dataTruth))   #the number of labels
prinCom = (torch.from_numpy(prinCom)).permute(1,2,0)
prinCom = flip(prinCom)
prinCom = (torch.from_numpy(prinCom)).permute(2,0,1)
datatruth = flip(dataTruth)

#to pick up the training and testing data
HalfWidth = 32
Wid = 2 * HalfWidth
G = datatruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
data = prinCom[:,nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
[row, col] = G.shape

NotZeroMask = np.zeros([row, col])
NotZeroMask[HalfWidth + 1: -1 - HalfWidth + 1, HalfWidth + 1: -1 - HalfWidth + 1] = 1
G = G * NotZeroMask

[Row, Column] = np.nonzero(G)
nSample = np.size(Row)

imdb = {}
imdb['trainData'] = np.zeros([nTrain, nBand, 2 * HalfWidth, 2 * HalfWidth], dtype=np.float32)
imdb['trainLabels'] = np.zeros([nTrain], dtype=np.int64)
imdb['testData'] = np.zeros([nTest, nBand, 2 * HalfWidth, 2 * HalfWidth], dtype=np.float32)
imdb['testLabels'] = np.zeros([nTest], dtype=np.int64)

RandPerm = np.random.permutation(nSample)

for iSample in range(nTrain):
    imdb['trainData'][iSample, :, :, :] = data[:,Row[RandPerm[iSample]] - HalfWidth: Row[RandPerm[iSample]] + HalfWidth, \
                                     Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth]
    imdb['trainLabels'][iSample] = G[Row[RandPerm[iSample]],
                                Column[RandPerm[iSample]]].astype(np.int64)
for iSample in range(nTest):
    imdb['testData'][iSample, :, :, :] = data[:,Row[RandPerm[(iSample+nTrain)]] - HalfWidth: Row[RandPerm[(iSample+nTrain)]] + HalfWidth, \
                                     Column[RandPerm[(iSample+nTrain)]] - HalfWidth: Column[RandPerm[(iSample+nTrain)]] + HalfWidth]
    imdb['testLabels'][iSample] = G[Row[RandPerm[(iSample+nTrain)]],
                                Column[RandPerm[(iSample+nTrain)]]].astype(np.int64)
print('Data is READY.')
imdb['trainLabels'] = imdb['trainLabels'] - 1
imdb['testLabels'] = imdb['testLabels'] - 1

#=============3D GAN=============#
  
#=============3D GENERATOR=============#
class G(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(G, self).__init__()
        self.layer1 = nn.ConvTranspose2d(nz, ngf*8, 4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*8)
        self.layer2 = nn.ConvTranspose2d(ngf*8, ngf*4, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*4)
        self.layer3 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*2)
        self.layer4 = nn.ConvTranspose2d(ngf*2, ngf, 4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.layer5 = nn.ConvTranspose2d(ngf, nc, 4, stride=2, padding=1, bias=False)
        self.ac = nn.ReLU(True)
        self.ac1 = nn.Tanh()

    def forward(self, z):
        #1024*1*1 to 512*4*4
        op = self.layer1(z)
        op = self.bn1(op)
        op = self.ac(op)
        
        #512*4*4 to 256*8*8
        op = self.layer2(op)
        op = self.bn2(op)
        op = self.ac(op)
        
        #256*8*8 to 128*16*16
        op = self.layer3(op)
        op = self.bn3(op)
        op = self.ac(op)
    
        #128*16*16 to 64*32*32
        op = self.layer4(op)
        op = self.bn4(op)
        op = self.ac(op)
    
        #64*32*32 to 3*64*64
        op = self.layer5(op)
        output = self.ac1(op)
        return output

#=============3D DISCRIMINATOR=============#    
class D(nn.Module):
    def __init__(self, ngf, nc, numLabel):
        super(D, self).__init__()
        self.layer1 =  nn.Conv2d(nc, ngf, 4, stride=2, padding=1, bias=False)
        self.layer2 = nn.Conv2d(ngf, ngf*2, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*2)
        self.layer3 =  nn.Conv2d(ngf*2, ngf*4, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*4)
        self.layer4 = nn.Conv2d(ngf*4, ngf*8, 4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf*8)
        self.layer5 = nn.Conv2d(ngf*8, ngf, 4, stride=1, padding=0, bias=False)
        self.ac = nn.LeakyReLU(0.2,inplace=True)
        self.ac1 = nn.Linear(ngf, 1)
        self.ac2 = nn.Linear(ngf, numLabel)
        self.layer6 = nn.Sigmoid()
        self.layer7 = nn.LogSoftmax()
        
    def forward(self, g):
        #3*64*64 to 64*32*32
        op = self.layer1(g)
        op = self.ac(op)

        #64*32*32 to 128*16*16
        op = self.layer2(op)
        op = self.bn2(op)
        op = self.ac(op)
 
        #128*16*16 to 256*8*8
        op = self.layer3(op)
        op = self.bn3(op)
        op = self.ac(op)
        
        #256*8*8 to 512*4*4
        op = self.layer4(op)
        op = self.bn4(op)
        op = self.ac(op)
        
        #512*4*4 to 64*1*1
        op = self.layer5(op)
        op = op.view(-1, ngf)
        
        #to distinguish the real or the fake 
        RorF = self.ac1(op)
        RorF = self.layer6(RorF)
        #to predict the class labels
        Classes = self.ac2(op)
        Classes = self.layer7(Classes)
        return RorF,Classes
    
#to compute the kappa
def kappa(testData, k):
    dataMat = np.mat(testData)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe  = float(ysum * xsum) / np.sum(dataMat) ** 2
    P0 = float(P0 / np.sum(dataMat) * 1.0)   #OA
    cohens_coefficient = float((P0 - Pe) / (1 - Pe))
    return cohens_coefficient

input = torch.FloatTensor(batSize, 3, 64, 64)
z = torch.FloatTensor(batSize, 100, 1, 1)
s_label = torch.FloatTensor(batSize)
c_label = torch.LongTensor(batSize)
inLabel= torch.LongTensor(nTest)
lossDD = np.zeros(50)
lossGG = np.zeros(50)
OA = np.zeros(50)
AA = np.zeros(50)
K = np.zeros(50)

real_label = 1
fake_label = 0
        
SCrit = nn.BCELoss()   #use Binary Cross Entropy Loss to compute the TorF loss
CCrit = nn.NLLLoss()   #use Negative Log Likelihood Loss to compute the HSI loss 
        
z = Variable(z)
input = Variable(input)
s_label = Variable(s_label)
c_label = Variable(c_label)
inLabel = Variable(inLabel)
G = G(nz, ngf, nc)
D = D(ngf, nc, numLabel)

optimizerD = optim.Adam(D.parameters(), lr=0.0002)   #to optimize with the Adam algorithm
optimizerG = optim.Adam(G.parameters(), lr=0.0002)

def train():
    plt.ion()   #to plot continously 
    for epoch in range(traEpo):
        #to train discriminator with fake data
        optimizerD.zero_grad()   #to clear the gradients of all optimized torch.Tensor s.
        z = torch.rand(batSize,100,1,1)  #to input the noise
        c = torch.randint(0, 16, (batSize,))     #to input the label
        gen = G(z)
        fakeSP,fakeCP = D(gen.detach())   #to input the fake data into discriminator 
        s_label.data.fill_(fake_label)
        c_label.copy_(c)
        lossDFS = SCrit(fakeSP, s_label)
        lossDFC = CCrit(fakeCP, c_label)
        lossDF = lossDFS + lossDFC
        lossDF.backward()
        
        #to train discriminator with REAL data 
        realSP,realCP = D(torch.from_numpy(imdb['trainData']))    #to input the real data into discriminator   
        s_label.fill_(real_label)
        c_label.copy_(torch.from_numpy(imdb['trainLabels']))           
        lossDRS = SCrit(realSP, s_label)
        lossDRC = CCrit(realCP, c_label)
        lossDR = lossDRS + lossDRC
        lossDR.backward()
        lossD = lossDR + lossDF   
        optimizerD.step()

        #to train generator
        optimizerG.zero_grad()
        fakeSP,fakeCP = D(gen)     #to input the fake data into discriminator   
        c_label.copy_(c)
        lossGS = SCrit(fakeSP, s_label)
        lossGC = CCrit(fakeCP, c_label)
        lossG = lossGS + lossGC   
        lossG.backward()
        optimizerG.step()

        #to output the result every 10 trainings
        if epoch % 10 == 0:  
             print('[%d]  Loss_D: %.4f Loss_G: %.4f '
                      % (epoch, lossD.data.numpy(), lossG.data.numpy()))
             lossDD[int(epoch/10)] = lossD / 10
             lossGG[int(epoch/10)] = lossG / 10
             
        #testing……     
        if epoch % 10 == 0:
            D.eval()
            G.eval()
            testLoss = 0
            right = 0
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)
            
            inLabel.copy_(torch.from_numpy(imdb['testLabels']))
            data, label = Variable(torch.from_numpy(imdb['testData'])), Variable(torch.from_numpy(imdb['testLabels']))
            y_l = label.data.cpu().numpy()
            output = D(data)
            testLoss += CCrit(output[1], label)
            pred = output[1].data.max(1)[1]  # get the index of the max log-probability
            right += pred.cpu().eq(inLabel).sum()
            predict = np.append(predict, pred.cpu().numpy())
            labels = np.append(labels, y_l)
 
            acc =100. * right / 500
            C = confusion_matrix(labels, predict)
            k = 100. * kappa(C, np.shape(C)[0])
            AA_ACC = np.diag(C) / np.sum(C, 1)
            aa = 100. * np.mean(AA_ACC, 0)
            print('OA= %.5f AA= %.5f k= %.5f' % (acc, aa, k))
            OA[int(epoch/10)] = acc
            AA[int(epoch/10)] = aa
            K[int(epoch/10)] = k
            
    #to plot the score of loss
    plt.figure(figsize=(7,10))       
    plt.subplot(2,1,1)
    x = np.arange(0,500,10) 
    plt.plot(x, lossDD, c='k', label='lossD')
    plt.plot(x, lossGG, c='r', label='lossG')
    plt.title('score of loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0,1)
    plt.legend(['lossD', 'lossG'])
    
    #to plot the score of OA、AA、kappa
    plt.subplot(2,1,2)
    plt.plot(x, OA, c='k', label='OA')
    plt.plot(x, AA, c='r', label='AA')
    plt.plot(x, K, c='b', label='kappa')
    plt.title('score of OA/AA/kappa')
    plt.xlabel('epoch')
    plt.ylabel('accuracy(%)') 
    plt.ylim(0,100)
    plt.legend(['OA', 'AA', 'kappa'])
    
    plt.show()
             
        
t_begin = time.time()             
train()
print("Total Elapse: {:.2f}".format(time.time() - t_begin))