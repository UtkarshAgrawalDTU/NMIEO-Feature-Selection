#[2016]-"The whale optimization algorithm"]

import numpy as np
from numpy.random import rand
from algos.functionHO import Fun
import math

def init_position(N, dim):
    X = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if rand() > 0.5:
                X[i,d] = 1;
    return X


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin


def tournament(fitness, param):
    Xt = fitness
    N = fitness.shape[0]
    Xt = Xt.reshape(N)
    
    Xt = [1/x for x in Xt]
    
    i1 = int(np.fix(rand()*N))
    i2 = int(np.fix(rand()*N))
    
    if i1 >= N:
        i1 = N-1
    if i2 >= N:
        i2 = N-1
        
    r = rand()
    if r < param:
        if Xt[i1] > Xt[i2]:
            return i1
        else:
            return i2
    else:
        if Xt[i1] > Xt[i2]:
            return i2
        else:
            return i1
    



def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x


def mutation(X, MR):
    dim = len(X)
    for d in range(dim):
        if rand() < MR:
            X[d] = 1 - X[d]

    return X 


def crossover(X1, X2): 
    dim = len(X1)
                  
    x1 = np.zeros([1, dim], dtype='int')
    x2 = np.zeros([1, dim], dtype='int')
    index   = np.random.randint(low = 1, high = dim-1)
    x1[0,:] = np.concatenate((X1[0:index] , X2[index:]))
    x2[0,:] = np.concatenate((X2[0:index] , X1[index:]))
    
    if rand() >= 0.5:
        return x1.reshape(dim)
    else:
        return x2.reshape(dim)

    
def woa_cm(xtrain, ytrain, opts):
    b     = 1       # constant
    N        = opts['N']
    max_iter = opts['T']
    if 'b' in opts:
        b    = opts['b']
    
    dim = np.size(xtrain, 1)
    X    = init_position(N, dim)
    
    fit  = np.zeros([N, 1], dtype='float')
    Xgb  = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
        
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    ans = {'fitness' : 1, 'acc': 0, 'num_feat': 0}
    
    for i in range(N):
        temp = Fun(xtrain, ytrain, X[i,:], opts)
        fit[i,0] = temp['fitness']
        if fit[i,0] < fitG:
            Xgb[0,:] = X[i,:]
            fitG     = fit[i,0]
            ans = temp
            
    curve[0,t] = fitG.copy()
    t += 1
 
    while t < max_iter:
        a = 2 - t * (2 / max_iter)
        MR = 0.9 + -0.9*(t-1)/(max_iter-1)
        
        for i in range(N):
            A = 2 * a * rand() - a
            C = 2 * rand()
            p = rand()
            if p  < 0.5:
                if abs(A) < 1:
                    X_temp = Xgb.reshape(dim)
                    X_mut = mutation(X_temp, MR)
                    Xc = crossover(X_mut, X[i,:])
                    X[i,:] = Xc
                    
                elif abs(A) >= 1:
                    k      = tournament(fit, 0.5)
                    X_rand = X[k, :]
                    X_mut = mutation(X_rand, MR)
                    Xc = crossover(X_rand, X[i, :])
                    X[i,:] = Xc

            
            elif p >= 0.5:
                for d in range(dim):
                    l = np.random.uniform(-1,1)
                    dist   = abs(Xgb[0,d] - X[i,d])
                    X[i,d] = dist * np.exp(b * l) * np.cos(2 * np.pi * l) + Xgb[0,d]
            
        
        for i in range(N):
            for d in range(dim):
                X[i,d] = boundary(X[i,d], 0, 1)
                    
        X = binary_conversion(X, 0.5, N, dim)
        
        for i in range(N):
            temp = Fun(xtrain, ytrain, X[i,:], opts)
            fit[i,0] = temp['fitness']
            if fit[i,0] < fitG:
                Xgb[0,:] = X[i,:]
                fitG     = fit[i,0]
                ans = temp
        
        curve[0,t] = fitG.copy()
        t += 1            

    Gbin       = Xgb
    Gbin       = Gbin.reshape(dim)    
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    
    ans['c'] = curve
    ans['sf'] = sel_index
    
    return ans 
