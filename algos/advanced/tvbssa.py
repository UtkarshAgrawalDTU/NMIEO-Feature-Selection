import numpy as np
from numpy.random import rand
from algos.functionHO import Fun


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
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


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x


def transfer_function(x):
    Tx = 1 / (1 + np.exp(-x))
    
    return Tx


def tvbssa(xtrain, ytrain, opts):
    # Parameters
    ub    = 1
    lb    = 0
    thres = 0.5
    
    N        = opts['N']
    max_iter = opts['T']
    
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    X     = init_position(lb, ub, N, dim)
    
    X     = binary_conversion(X, thres, N, dim)
    ans = {'fitness' : 1, 'acc': 0, 'num_feat': 0}

    
    fit   = np.zeros([N, 1], dtype='float')
    Xf    = np.zeros([1, dim], dtype='int')
    fitF  = float('inf')
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0

    while t < max_iter:
        
        for i in range(N):
            temp = Fun(xtrain, ytrain, X[i,:], opts)
            fit[i,0] = temp['fitness']
            if fit[i,0] < fitF:
                Xf[0,:] = X[i,:]
                fitF    = fit[i,0]
                ans = temp
        
        curve[0,t] = fitF.copy()
        t += 1
        
        c1 = 2 * np.exp(-(4 * t / max_iter) ** 2)
        L  = np.ceil(N * (t / (max_iter + 1))) 
        
        for i in range(N):          
            if i < L:  
                for d in range(dim):
                    c2 = rand() 
                    c3 = rand()
                    if c3 >= 0.5: 
                        Xn = Xf[0,d] + c1 * ((ub[0,d] - lb[0,d]) * c2 + lb[0,d])
                    else:
                        Xn = Xf[0,d] - c1 * ((ub[0,d] - lb[0,d]) * c2 + lb[0,d])
                
                    Tx = transfer_function(Xn)
                    if rand() < Tx:
                        X[i,d] = 0
                    else:
                        X[i,d] = 1
                
            else:
                for d in range(dim):
                    Xn = (X[i,d] + X[i-1, d]) / 2
                    Xn = boundary(Xn, lb[0,d], ub[0,d]) 
                    if Xn > thres:
                        X[i,d] = 1
                    else:
                        X[i,d] = 0


    Gbin       = Xf[0,:] 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    ans['c'] = curve
    ans['sf'] = sel_index
    return ans  