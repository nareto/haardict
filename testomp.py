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
        
np.random.seed(234234430)
n,m = 100,512
tolerance = 1e-10
sparsity = 200
normalize = False
sigma=0.1
#normalize = True

#ocdict = np.random.uniform(size=(n,m))
ocdict = np.random.normal(scale=sigma,size=(n,m))
y = np.random.uniform(size=(n,))
#ocdict = np.arange(16).reshape(4,4)
#y = np.array([1,0,0,0])
mean = np.mean(y)
#y -= mean
if normalize:
    y /= np.linalg.norm(y)
    for col in range(m):
        ocdict[:,col] /= np.linalg.norm(ocdict[:,col])

#print(y)


#omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity,fit_intercept=True,normalize=True)
##omp = OrthogonalMatchingPursuit(fit_intercept=True,normalize=True,tol=tolerance)
##omp = OrthogonalMatchingPursuitCV(fit_intercept=True,normalize=True)
##omp = OrthogonalMatchingPursuit(fit_intercept=True,normalize=True)
##omp = OrthogonalMatchingPursuit(tol=tolerance)
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)#,tol=tolerance)
##omp = OrthogonalMatchingPursuit()
omp.fit(ocdict,y)
coef = omp.coef_



#y = y.astype('float64')
#ocdict = ocdict.astype('float64')
#coef = octave.OMP(sparsity,y,ocdict).transpose()


#out = np.zeros(shape=(n,))
out = np.zeros_like(y,dtype='float64')
for idx in coef.nonzero()[0]:
    out += coef[idx]*ocdict[:,idx]
out2 = np.dot(ocdict,coef)
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


print('Number of coefficients used: %d\nError in sparse coding: %f\nOut2: %f' %\
      (len(coef.nonzero()[0]),np.linalg.norm(out - y), np.linalg.norm(out2-y)))
