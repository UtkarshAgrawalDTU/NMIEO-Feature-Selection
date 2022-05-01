import numpy as np
from numpy.random import rand
from algos.functionHO import Fun
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.cluster import normalized_mutual_info_score
import random

def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        X[i,0] = rand()
        for d in range(1,dim):
#             X[i,d] = rand()
            X[i,d] = 4*X[i,d-1]*(1-X[i,d-1])
    return X


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        return lb
    if x > ub:
        return ub
    return x


def local_search(Xgb, Xl, dim):
        
    sel_indexes = []
    notsel_indexes = []

    Xgb_bin = binary_conversion(Xgb, 0.5, 1, dim)
    Xgbf = Xgb_bin
    neigh = NearestNeighbors(n_neighbors= 10)
    neigh.fit(Xl)
    
    
    for d in range(dim):
        if Xgb_bin[0, d] == 0:
            notsel_indexes.append(d)
        else:
            sel_indexes.append(d)
    
    if len(sel_indexes) >= 2:
        index_list = random.sample(range(0, len(sel_indexes)), 2)
        index1d = index_list[0]
        index2d = index_list[1]
                
        score1 = []
        n1 = neigh.kneighbors([Xl[index1d]], 10, return_distance=False)
        for i in n1[0]:
            s = normalized_mutual_info_score(Xl[i], Xl[index1d])
            score1.append(s)
        score1d = np.mean(np.asarray(score1)) 

        n2 = neigh.kneighbors([Xl[index2d]], 10, return_distance=False)
        score2 = []

        for i in n2[0]:
            s = normalized_mutual_info_score(Xl[i], Xl[index2d])
            score2.append(s)
        score2d = np.mean(np.asarray(score2))

        if score1d > score2d:
            Xgbf[0, index1d] = 0
        else:
            Xgbf[0, index2d] = 0

    if len(notsel_indexes) >= 2:
        notsel_index_list = random.sample(range(0, len(notsel_indexes)), 2)
        index1a = notsel_index_list[0]
        index2a = notsel_index_list[1]
                                         
        score1 = []
        n1 = neigh.kneighbors([Xl[index1a]], 10, return_distance=False)
        for i in n1[0]:
            s = normalized_mutual_info_score(Xl[i], Xl[index1a])
            score1.append(s)

        score1a = np.mean(np.asarray(score1)) 

        n2 = neigh.kneighbors([Xl[index2a]], 10, return_distance=False)
        score2 = []

        for i in n2[0]:
            s = normalized_mutual_info_score(Xl[i], Xl[index2a])
            score2.append(s)
        score2a = np.mean(np.asarray(score2))

        if score1a < score2a:
            Xgbf[0, index1a] = 1
        else:
            Xgbf[0, index2a] = 1

    return Xgbf


def nmieo(xtrain, ytrain, opts):
    ub     = 1
    lb     = 0
    thres  = 0.5
    
    a1    = 2;     
    a2    = 1;     
    GP    = 0.5;   
    V     = 1;     
    
    N          = opts['N']
    max_iter   = opts['T']
        
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    X     = init_position(lb, ub, N, dim)
    Xl = np.transpose(xtrain)
    
    Xmb = np.zeros([N, dim], dtype='float')
    fitM   = np.ones([N, 1], dtype='float')
    
    fitE1  = float('inf')
    fitE2  = float('inf')
    fitE3  = float('inf')
    fitE4  = float('inf')
    fitE5  = float('inf')
    fitE6  = float('inf')
    fitE7  = float('inf')
    fitE8  = float('inf')
    
    
    Xeq1   = np.zeros([1, dim], dtype='float')
    Xeq2   = np.zeros([1, dim], dtype='float')
    Xeq3   = np.zeros([1, dim], dtype='float')
    Xeq4   = np.zeros([1, dim], dtype='float')
    Xeq5   = np.zeros([1, dim], dtype='float')
    Xeq6   = np.zeros([1, dim], dtype='float')
    Xeq7   = np.zeros([1, dim], dtype='float')
    Xeq8   = np.zeros([1, dim], dtype='float')
    Xave   = np.zeros([1, dim], dtype='float')
    fit   = np.zeros([N, 1], dtype='float')
        
    ans = {'fitness' : 1, 'acc': 0, 'num_feat': 0}
    
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    
    r1 = rand()
    r2 = rand()
    lamb = rand()
    r = rand()
    while t < max_iter:
        
        Xbin = binary_conversion(X, thres, N, dim)
        
        for i in range(N):
            temp = Fun(xtrain, ytrain, Xbin[i,:], opts)
            fit[i, 0] = temp['fitness']
            if fit[i, 0] < fitE1:
                fitE1 = fit[i, 0]
                Xeq1[0,:] = X[i, :]
                ans = temp
            elif fit[i, 0] < fitE2:
                fitE2 = fit[i, 0]
                Xeq2[0,:]  = X[i, :]
            elif fit[i, 0] < fitE3:
                fitE3 = fit[i, 0]
                Xeq3[0,:]  = X[i, :]
            elif fit[i, 0] < fitE4:
                fitE4 = fit[i, 0]
                Xeq4[0,:]  = X[i, :]
            elif fit[i, 0] < fitE5:
                fitE5 = fit[i, 0]
                Xeq5[0,:]  = X[i, :]
            elif fit[i, 0] < fitE6:
                fitE6 = fit[i, 0]
                Xeq6[0,:]  = X[i, :]
            elif fit[i, 0] < fitE7:
                fitE7 = fit[i, 0]
                Xeq7[0,:]  = X[i, :]
            elif fit[i, 0] < fitE8:
                fitE8 = fit[i, 0]
                Xeq8[0,:]  = X[i, :]
                
        for i in range(N):
            if fitM[i,0] < fit[i,0]:
                fit[i,0] = fitM[i,0]
                X[i, :]  = Xmb[i, :]
                
        Xmb = X
        fitM = fit
        
        for d in range(dim):
            Xave[0,d] = (Xeq1[0,d] + Xeq2[0,d] + Xeq3[0,d] + Xeq4[0,d] + Xeq5[0,d] + Xeq6[0,d] + Xeq7[0,d] + Xeq8[0,d])/8
        
        Xpool = np.asarray([Xeq1, Xeq2, Xeq3, Xeq4, Xeq5, Xeq6, Xeq7, Xeq8, Xave])
        
        T     = (1 - (t / max_iter)) ** (a2 * (t / max_iter))
        
        for i in range(N):
            r1 = 4*r1*(1-r1)
            r2 = 4*r2*(1-r2)
            
            if r2 >= GP:
                GCP = 0.5*r1
            else:
                GCP = 0
            eq = np.random.randint(0,8)
            
            for d in range(dim):
                r = 4*r*(1-r)
                lamb = 4*lamb*(1-lamb)
                F  = a1 * np.sign(r - 0.5) * (np.exp(-lamb * T) - 1)
                G0 = GCP * (Xpool[eq][0,d] - lamb * X[i,d])
                G  = G0 * F
      
                X[i,d] = Xpool[eq][0,d] + (X[i,d] - Xpool[eq][0,d])*F + (G/(lamb * V))*(1 - F)
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
        
        Xeq1t = local_search(Xeq1, Xl, dim)
        temp = Fun(xtrain, ytrain, Xeq1t[0,:], opts)
        if temp['fitness'] < fitE1:
                Xeq1[0,:]   = Xeq1t[0,:]
                ans = temp
        
        curve[0,t] = fitE1.copy()
        t += 1            

    Gbin       = binary_conversion(Xeq1, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    
    ans['c'] = curve
    ans['sf'] = sel_index
    return ans  


            
            
                