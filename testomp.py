from oct2py import octave
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
#import queue

def mysnorm(x):
    ret = 0
    for xi in x:
        ret += xi**2
    return(ret)

def plot_sparsities():
    #np.random.seed(234234430)
    max_sparsity = 99
    sparsities = np.arange(1,max_sparsity)
    values = -np.ones_like(sparsities,dtype='float64')
    for idx,s in np.ndenumerate(sparsities):
        rank,spars,v = simple_test(s,plot=False,outprint=False)
        values[idx] = v

    print('dict rank: %f' % rank)
    fig = plt.figure()
    ax = fig.gca()
    ax.set_ylim(values.min(),values.max())
    plt.plot(sparsities,values)
    plt.show()
    return(values)

def simple_test(sparsity=50,plot=False,outprint=True):        
    np.random.seed(234234430)
    n,m = 100,512
    reshape = (10,10)
    tolerance = 1e-10
    sigma=0.1
    avg = 0
    #normalize = True
    normalize = False

    #ocdict = np.random.uniform(size=(n,m))
    ocdict = np.random.normal(loc=avg,scale=sigma,size=(n,m))
    y = np.random.uniform(size=(n,))
    rank = np.linalg.matrix_rank(ocdict)
    #ocdict = np.arange(16).reshape(4,4)
    #y = np.array([1,0,0,0])
    mean = np.mean(y)
    #print(mean)
    y -= mean
    if normalize:
        y /= np.linalg.norm(y)
        for col in range(m):
            #ocdict[:,col] /= np.linalg.norm(ocdict[:,col])
            ocdict /= np.linalg.norm(ocdict)

    #print(y)


    #omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity,fit_intercept=True,normalize=True)
    ##omp = OrthogonalMatchingPursuit(fit_intercept=True,normalize=True,tol=tolerance)
    ##omp = OrthogonalMatchingPursuitCV(fit_intercept=True,normalize=True)
    ##omp = OrthogonalMatchingPursuit(fit_intercept=True,normalize=True)
    ##omp = OrthogonalMatchingPursuit(tol=tolerance)
    #omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity,fit_intercept=True,normalize=True)#,tol=tolerance)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
    omp.fit(ocdict,y)
    coef = omp.coef_



    #y = y.astype('float64')
    #ocdict = ocdict.astype('float64')
    #coef = octave.OMP(sparsity,y,ocdict).transpose()


    #out = np.zeros(shape=(n,))
    #out = np.zeros_like(y,dtype='float64')
    #for idx in coef.nonzero()[0]:
    #    out += coef[idx]*ocdict[:,idx]
    out = np.dot(ocdict,coef)
    #out += mean
    #score = omp.score(ocdict,y)
    #score= omp.score(coef,y)
    #myuv = (np.linalg.norm(y-out)/np.linalg.norm(y - mean))**2
    #myuv = mysnorm(y-out)/mysnorm(y-mean)
    #myscore = 1 - myuv
    #print('Number of coefficients used: %d\nError in sparse coding: %f\nCoefficient of determination: %f\nu/v: %f\nmy score: %f\nmy u/v %f' %\
    #      (len(coef.nonzero()[0]),np.linalg.norm(out - y),score,1-score,myscore,myuv))
    #print('Number of coefficients used: %d\nError in sparse coding: %f' %\
    #      (len(coef.nonzero()[0]),np.linalg.norm(out - y)))


    #print('Number of coefficients used: %d\nError in sparse coding: %f\nOut2: %f' %\
  #        (len(coef.nonzero()[0]),np.linalg.norm(out - y), np.linalg.norm(out2-y)))
    out += mean
    y += mean
    err = np.linalg.norm(out - y)
    spars = len(coef.nonzero()[0])
    if outprint:
        print('Number of coefficients used: %d\nError in sparse coding: %f\n' %\
              (spars,err))
    if plot:
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(y.reshape(reshape), cmap=plt.cm.gray,interpolation='none')
        ax2.imshow(out.reshape(reshape), cmap=plt.cm.gray,interpolation='none')
        fig.show()
    return((rank,spars,err))
