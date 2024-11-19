import math
import scipy
import scipy.linalg 
import numpy as np
import json
import os
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

##Use proper function to load SNP, should be NA filled and centered
def imputeY(snp, beta, batch_size = 20000):
    ##value of lambda, could be changed to  your own choice
    lam=1e-6
    test_n = snp.shape[0]
    num_batches = int(np.ceil(test_n/batch_size))
    p = snp.shape[1]
    #
    yhat = np.zeros(test_n)
    time1=time.process_time()
    for i in range(num_batches):
        start = i*batch_size
        end = min(test_n, (i+1)*batch_size)
        snp_batch = snp[start:end,:]
        snp_batch = snp_batch - snp_batch.mean(axis=0) 
        D = np.diag(snp_batch.T@snp_batch)
        D = np.diag(1/D)
        tmpM = D@snp_batch.T
        xxt = tmpM.T@tmpM
        a1 = np.diag(xxt)
        a2 = a1+lam
        np.fill_diagonal(xxt,a2)
        xxtinv=np.linalg.inv(xxt)
        tmp_yhat = xxtinv@tmpM.T@beta
        yhat[start:end] = tmp_yhat
    time2 = time.process_time()
    chtime = time2-time1
    ##use proper function to save the result
    return yhat, chtime

def imputeY_batch(snp_batch, beta):
    lam=1e-6
    time1=time.process_time()
    snp_batch = snp_batch - snp_batch.mean(axis=0)
    D = np.diag(snp_batch.T@snp_batch)
    D = np.diag(1/D)
    tmpM = D@snp_batch.T
    xxt = tmpM.T@tmpM
    a1 = np.diag(xxt)
    a2 = a1+lam
    np.fill_diagonal(xxt,a2)
    xxtinv=np.linalg.inv(xxt)
    tmp_yhat = xxtinv@tmpM.T@beta
    time2 = time.process_time()
    chtime = time2-time1
    return tmp_yhat, chtime

def imputeY_batch_cholesky(snp_batch, beta):
    lam=1e-6
    time1=time.process_time()
    snp_batch = snp_batch - snp_batch.mean(axis=0)
    D = np.diag(snp_batch.T@snp_batch)
    D = np.diag(1/D)
    tmpM = D@snp_batch.T
    xxt = tmpM.T@tmpM
    a1 = np.diag(xxt)
    a2 = a1+lam
    np.fill_diagonal(xxt,a2)
    xtbeta=np.matmul(tmpM.T,beta)
    u= scipy.linalg.cholesky(xxt)
    w = scipy.linalg.solve(u.T, xtbeta)
    tmp_yhat=scipy.linalg.solve(u,w)
    time2 = time.process_time()
    chtime = time2-time1
    return tmp_yhat, chtime