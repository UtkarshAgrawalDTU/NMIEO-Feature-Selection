import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from statistics import mean


# error rate
def error_rate(X_train, y_train, X_test, y_test, x, opts):

    k     = opts['k']    
    num_train = np.size(X_train, 0)
    num_valid = np.size(X_test, 0)
    xtrain  = X_train[:, x == 1]
    ytrain  = y_train.reshape(num_train) 
    xvalid  = X_test[:, x == 1]
    yvalid  = y_test.reshape(num_valid)
    mdl     = KNeighborsClassifier(n_neighbors = k, p = 2)
    mdl.fit(xtrain, ytrain)
    ypred   = mdl.predict(xvalid)
    acc     = np.sum(yvalid == ypred) / num_valid
    error   = 1 - acc
    return error


def Fun(X, y, x, opts):

    alpha    = 0.99
    beta     = 1 - alpha
    
    cost_avg = []
    acc_avg = []
    num_feat_avg = []
    
    kf = ShuffleSplit(n_splits = 10)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        max_feat = len(x)
        num_feat = np.sum(x == 1)
        if num_feat == 0:
            cost  = 1
            acc = 0
        else:
            error = error_rate(X_train, y_train, X_test, y_test, x, opts)
            cost  = alpha * error + beta * (num_feat / max_feat)
            acc = 1-error
        cost_avg.append(cost)
        acc_avg.append(acc)
        num_feat_avg.append(float(num_feat))
        
    return {'fitness' : mean(cost_avg), 'acc': mean(acc_avg), 'num_feat': mean(num_feat_avg)}

