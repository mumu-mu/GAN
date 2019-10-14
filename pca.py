# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:43:15 2019

@author: lenovo
"""
import scipy.io as sio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


#turn to float
def confloat(x):
    r=[float(i) for i in x]
    return r

#=============PCA=============#

def pca(dataMat,topNfeat=9999999):
    #dataMat：input
    #topNfeat：output top N features
    
    meanVals = np.mean(dataMat,axis=0)  #to compute the mean value
    meanRemoved = dataMat - meanVals  #to remove the mean value
    covMat = np.cov(meanRemoved,rowvar=0)   #to compute the covariance matrix
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))  #to compute the eigenvalues and eigenvectors
    eigValInd = np.argsort(eigVals)   #to sort from the smallest to the largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #to get the nth largest eigenvalues and eigenvectors
    redEigVects=eigVects[:,eigValInd]
    lowDDataMat = meanRemoved * redEigVects   #to transform into low dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    cumCont=sum(eigVals[eigValInd])/sum(eigVals)   #to compute cumulative contribution
    return lowDDataMat,reconMat,cumCont

#load the data
dataF='C://Users//lenovo//Desktop//毕设//资料//Salinas_corrected.mat'
data = sio.loadmat(dataF)["salinas_corrected"]
imData= data.reshape([111104,204])  #reshape the data into 2_D
lowDataMat,reconMat,cumCont=pca(imData,3)
print(cumCont)   #cumulative contribution
prinCom1 =lowDataMat[:,0].reshape([512,217])   #principal component 1
prinCom2 =lowDataMat[:,1].reshape([512,217])    #principal component 2
prinCom3 =lowDataMat[:,2].reshape([512,217])    #principal component 3
#prinCom = np.array([prinCom1,prinCom2,prinCom3])      #principal component_merge
#print(lowDataMat)

import matplotlib    #to show row data and data in low dimension
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(imData[:,0].flatten(),imData[:,1].flatten(),s=10)
ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],s=10,c='red')
plt.show()



#plt.imshow(data)