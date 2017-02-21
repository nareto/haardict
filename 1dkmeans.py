import matplotlib.pyplot as plt
import numpy as np


def centroid(S):
    if S.size == 0:
        raise Exception("Can't compute centroid of void set")
    centroid = 0
    for val in S:
        centroid += val
    centroid /= S.size
    return(centroid)

def twomeansval(S,k):
    C1 = S[:k]
    C2 = S[k:]
    c1bar = centroid(C1)
    c2bar = centroid(C2)
    val = 0
    for s in C1:
        val += (s-c1bar)**2
    for s in C2:
        val += (s-c2bar)**2
    return(val)
    
n=102
S = np.random.uniform(size=n)
#clust1 = np.random.normal(loc=2,scale=1,size=n/3)
#clust2 = np.random.normal(loc=20,scale=1,size=n/3)
#clust3 = np.random.normal(loc=200,scale=10,size=n/3)
#S = np.hstack((clust1,clust2,clust3))
minval = S.min()
maxval = S.max()
S.sort()
values = -np.ones_like(S[1:])
for k in range(1,n):
    values[k-1] = twomeansval(S,k)
values = np.hstack((values[0],values))
rescaledvalues = np.zeros_like(S)
for k in range(n):
    s = S[k]
    #rescaleds = n*s
    diff = maxval-minval
    rescaleds = s*n/diff - n*minval/diff
    rescaledvalues[k] = rescaleds
    plt.plot(rescaleds,0,'ro')

plt.plot(rescaledvalues,values,'-x')
plt.show()
