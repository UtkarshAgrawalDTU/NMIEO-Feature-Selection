#[2010]-"A new metaheuristic bat-inspired algorithm"

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
    

def bba(xtrain, ytrain, opts):
    
    fmax   = 2      # maximum frequency
    fmin   = 0      # minimum frequency
    alpha  = 0.9    # constant
    gamma  = 0.9    # constant
    A_max  = 2      # maximum loudness
    r0_max = 1      # maximum pulse rate
    
    N          = opts['N']
    max_iter   = opts['T']
    if 'fmax' in opts:
        fmax   = opts['fmax'] 
    if 'fmin' in opts:
        fmin   = opts['fmin'] 
    if 'alpha' in opts:
        alpha  = opts['alpha'] 
    if 'gamma' in opts:
        gamma  = opts['gamma'] 
    if 'A' in opts:
        A_max  = opts['A'] 
    if 'r' in opts:
        r0_max = opts['r'] 
        
    dim = np.size(xtrain, 1)
        
    X     = init_position(N, dim)
    V     = np.zeros([N, dim], dtype='float')
    
    fit   = np.zeros([N, 1], dtype='float')
    Xgb   = np.zeros([1, dim], dtype='int')
    fitG  = float('inf')
    
    ans = {'fitness' : 1, 'acc': 0, 'num_feat': 0}
    
    for i in range(N):
        temp = Fun(xtrain, ytrain, X[i,:], opts)
        fit[i,0] = temp['fitness'] 
        if fit[i,0] < fitG:
            Xgb[0,:] = X[i,:]
            fitG     = fit[i,0]
    
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    A  = np.random.uniform(1, A_max, N)
    r0 = np.random.uniform(0, r0_max, N)
    r  = r0.copy()
    
    while t < max_iter:
        for i in range(N):
            for d in range(dim):
                freq = fmin + (fmax - fmin) * rand()
                
                V[i,d] = V[i,d] + (X[i,d] - Xgb[0,d])*freq 
                TF = abs((2/math.pi)*math.atan((math.pi/2)*V[i,d]))
                
                if rand() < TF:
                    if X[i,d] == 0:
                        X[i,d] = 1
                    else:
                        X[i,d] = 0
                else:
                    X[i,d] = X[i,d]
                    
                if rand() > r[i]:
                    X[i,d] = Xgb[0,d]
            
            temp = Fun(xtrain, ytrain, X[i,:], opts)
            Fnew = temp['fitness']
            
            if Fnew <= fit[i,0] and rand() < A[i]:  
                X[i,:] = X[i,:]
                fit[i,0] = Fnew
                
            if Fnew <= fitG:
                Xgb[0,:] = X[i,:]
                fitG = Fnew
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

            
            
                