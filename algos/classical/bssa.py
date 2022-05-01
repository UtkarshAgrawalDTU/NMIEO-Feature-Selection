#[2017]-"Salp swarm algorithm: A bio-inspired optimizer for engineering design problems"

import numpy as np
from numpy.random import rand
from algos.functionHO import Fun
import math

def init_position(N, dim):
    X = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if rand() > 0.5:
                X[i,d] = 1
            else:
                X[i,d] = 0       
    
    return X


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x



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



def bssa(xtrain, ytrain, opts):
    
    N        = opts['N']
    max_iter = opts['T']
    dim = np.size(xtrain, 1)
    X     = init_position(N, dim)
    
    fit   = np.zeros([N, 1], dtype='float')
    Xf    = np.zeros([1, dim], dtype='int')
    fitF  = float('inf')
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    
    ans = {'fitness' : 1, 'acc': 0, 'num_feat': 0}
    
    while t < max_iter:
        
        c1 = 2 * np.exp(-(4 * t / max_iter) ** 2)
        
        for i in range(N):          
            if i <= N/2:  
                for d in range(dim):
                    c2 = rand() 
                    c3 = rand()
                    if c3 >= 0.5: 
                        X[i,d] = Xf[0,d] + c1*c2
                        TF = 1 / (1 + math.exp(-1*X[i,d]))
                        if TF > rand():
                            X[i,d] = 1
                        else:
                            X[i,d] = 0
                    else:
                        X[i,d] = Xf[0,d] - c1*c2
                        TF = 1 / (1 + math.exp(-1*X[i,d]))
                        if TF > rand():
                            X[i,d] = 1
                        else:
                            X[i,d] = 0
                
            elif i > N/2 and i < N+1:
                Xc = crossover(X[i, :], X[i-1, :])
                X[i,:] = Xc
         
        for i in range(N):
            for d in range(dim):
                X[i,d] = boundary(X[i,d], 0, 1)
            temp = Fun(xtrain, ytrain, X[i,:], opts)
            fit[i,0] = temp['fitness']
            if fit[i,0] < fitF:
                fitF  = fit[i,0]
                Xf[0,:]  = X[i,:]
                ans = temp
                
        curve[0,t] = fitF.copy()
        t += 1
        

    Gbin       = Xf
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    
    ans['c'] = curve
    ans['sf'] = sel_index
    
    return ans  