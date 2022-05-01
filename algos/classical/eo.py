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
    

def eo(xtrain, ytrain, opts):
    # Parameters
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
    
    Xmb = np.zeros([N, dim], dtype='float')
    fitM   = np.ones([N, 1], dtype='float')
    
    fitE1  = float('inf')
    fitE2  = float('inf')
    fitE3  = float('inf')
    fitE4  = float('inf')
    
    Xeq1   = np.zeros([1, dim], dtype='float')
    Xeq2   = np.zeros([1, dim], dtype='float')
    Xeq3   = np.zeros([1, dim], dtype='float')
    Xeq4   = np.zeros([1, dim], dtype='float')
    Xave   = np.zeros([1, dim], dtype='float')
    fit   = np.zeros([N, 1], dtype='float')
        
    ans = {'fitness' : 1, 'acc': 0, 'num_feat': 0}
    
    # Pre
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    
    while t < max_iter:
        
        Xbin = binary_conversion(X, thres, N, dim)
        
        for i in range(N):
            temp = Fun(xtrain, ytrain, Xbin[i,:], opts)
            fit[i, 0] = temp['fitness']
            if fit[i, 0] < fitE1:
                fitE1 = fit[i, 0]
                Xeq1[0,:] = X[i, :]
                ans = temp
            elif fit[i, 0] > fitE1 and fit[i, 0] < fitE2:
                fitE2 = fit[i, 0]
                Xeq2[0,:]  = X[i, :]
            elif fit[i, 0] > fitE1 and fit[i, 0] > fitE2 and fit[i, 0] < fitE3:
                fitE3 = fit[i, 0]
                Xeq3[0,:]  = X[i, :]
            elif fit[i, 0] > fitE1 and fit[i, 0] > fitE2 and fit[i, 0] > fitE3 and fit[i, 0] < fitE4:
                fitE4 = fit[i, 0]
                Xeq4[0,:]  = X[i, :]
                
                
        for i in range(N):
            if fitM[i,0] < fit[i,0]:
                fit[i,0] = fitM[i,0]
                X[i, :]  = Xmb[i, :]
                
        Xmb = X
        fitM = fit
        
        for d in range(dim):
            Xave[0,d] = (Xeq1[0,d] + Xeq2[0,d] + Xeq3[0,d] + Xeq4[0,d])/4
        
        Xpool = np.asarray([Xeq1, Xeq2, Xeq3, Xeq4, Xave])
        
        T     = (1 - (t / max_iter)) ** (a2 * (t / max_iter))
        
        for i in range(N):
            r1 = rand()
            r2 = rand()
            
            if r2 >= GP:
                GCP = 0.5*r1
            else:
                GCP = 0
            eq = np.random.randint(0,4)
            
            for d in range(dim):
                r = rand()
                lamb = rand()
                F  = a1 * np.sign(r - 0.5) * (np.exp(-lamb * T) - 1)
                G0 = GCP * (Xpool[eq][0,d] - lamb * X[i,d])
                G  = G0 * F
      
                X[i,d] = Xpool[eq][0,d] + (X[i,d] - Xpool[eq][0,d])*F + (G/(lamb * V))*(1 - F)
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
        
        # Store result
        curve[0,t] = fitE1.copy()
        t += 1            

            
    # Best feature subset
    Gbin       = binary_conversion(Xeq1, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    
    ans['c'] = curve
    ans['sf'] = sel_index
    return ans  


            
            
                