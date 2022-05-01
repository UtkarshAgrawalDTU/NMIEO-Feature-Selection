import numpy as np
from numpy.random import rand
from algos.functionHO import Fun
import math


def init_position(N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            if rand() > 0.5:
                X[i,d] = 1;
    return X


def init_velocity(N, dim):
    V    = np.zeros([N, dim], dtype='float')
    Vmax = np.zeros([1, dim], dtype='float')
    Vmin = np.zeros([1, dim], dtype='float')
    for d in range(dim):
        Vmax[0,d] = 6
        Vmin[0,d] = -6
        
    return V, Vmax, Vmin


def binary_conversion(V):
    
    TF = 1 / (1 + math.exp(-1*V))
    if TF > rand():
        return 1
    else:
        return 0


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x
    

def bpso(xtrain, ytrain, opts):
    # Parameters

    w     = 0.9    
    c1    = 2      
    c2    = 2     
    
    if 'w' in opts:
        w    = opts['w']
    if 'c1' in opts:
        c1   = opts['c1']
    if 'c2' in opts:
        c2   = opts['c2'] 
    
    N        = opts['N']
    max_iter = opts['T']
    
    dim = np.size(xtrain, 1)

    X = init_position(N, dim)
    V, Vmax, Vmin = init_velocity(N, dim)
        
    fit   = np.zeros([N, 1], dtype='float')
    Xgb   = np.zeros([1, dim], dtype='float')
    fitG  = float('inf')
    
    for i in range(N):
        temp = Fun(xtrain, ytrain, X[i,:], opts)
        fit[i,0] = temp['fitness']
        if fit[i,0] < fitG:
            fitG  = fit[i,0]
            Xgb[0,:]  = X[i,:]
    
    # Pre

    Xpb   = X
    fitP  = fit
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
 
    W = np.linspace(0.9, 0.4, max_iter)   
    ans = {'fitness' : 1, 'acc': 0, 'num_feat': 0}
    
    while t < max_iter:
        
        for i in range(N):
            for d in range(dim):
                r1     = rand()
                r2     = rand()
                VB = W[t] * V[i,d] + c1 * r1 * (Xpb[i,d] - X[i,d]) + c2 * r2 * (Xgb[0,d] - X[i,d]) 
                VB = boundary(VB, Vmin[0,d], Vmax[0,d])
                V[i,d] = VB
                X[i,d] = binary_conversion(V[i,d])
        
            temp = Fun(xtrain, ytrain, X[i,:], opts)
            fit[i,0] = temp['fitness']
            if fit[i,0] < fitP[i,0]:
                Xpb[i,:]  = X[i,:]
                fitP[i,0] = fit[i,0]
            if fitP[i,0] < fitG:
                Xgb[0,:]  = Xpb[i,:]
                fitG      = fitP[i,0]
            if temp['fitness'] < ans['fitness']:
                ans = temp
        
        curve[0,t] = fitG.copy()
        t += 1
    
    # Best feature subset
        
    Gbin       = Xgb 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    
    ans['c'] = curve
    ans['sf'] = sel_index
    
    return ans
    
    







