import numpy as np
import matplotlib.pyplot as plt
from recursivetwomeans import RecursiveTwoMeans
from kmaxoids import KMaxoids
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import time

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        toprint = ''
        if self.name:
            toprint += f'[Timer: {self.name}] '
        toprint += 'Elapsed: %s' % (time.time() - self.tstart)
        print(toprint)
            


def make_gauss_blob(center,pca1,var1=2,var2=0.5,npoints=500):
    points = []
    pca1 = np.array(pca1)
    center = np.array(center)
    for i in range(npoints):
        pca1_c = pca1*np.random.normal(0,var1)
        pca2 = np.array((pca1[1],-pca1[0]))
        pca2_c = pca2*np.random.normal(0,var2)
        p = center + pca1_c + pca2_c
        points.append(p)
    return(points)

l = 4
b1 = np.array(make_gauss_blob((l,l),(1,1)))
b2 = np.array(make_gauss_blob((-l,l),(-1,1)))
b3 = np.array(make_gauss_blob((0,-l),(0,-1)))
Y = np.vstack([b1,b2,b3])
#each row is a data point

fig,axes = plt.subplots(1,3)
colors = ['red','green','blue', 'yellow', 'purple', 'orange', 'pink']

#KMeans
with Timer('Kmeans'):
    kmeans = KMeans(n_clusters=3).fit(Y)
for p,coord in enumerate(Y):
    axes[0].scatter(coord[0],coord[1],color=colors[kmeans.labels_[p]])
for cent in kmeans.cluster_centers_:
    axes[0].plot(cent[0],cent[1],'ok')

#RecursiveTwoMeans
with Timer('RecursiveTwoMeans'):
    recursive2means = RecursiveTwoMeans(wcss_tol = 100, ncomponents=3).fit(Y,100)
for p,coord in enumerate(Y):
    axes[1].scatter(coord[0],coord[1],color=colors[recursive2means.labels_[p] % len(colors)])
for cent in recursive2means.cluster_centers_:
    axes[1].plot(cent[0],cent[1],'ok')
    
#KMaxoids
kmaxoids = KMaxoids(Y.transpose(),3)
with Timer('KMaxoids'):
    maxoids, labels = kmaxoids.run()
for p,coord in enumerate(Y):
    axes[2].scatter(coord[0],coord[1],color=colors[labels[p]])
#for p,c in enumerate(colors):
#    cent = maxoids[:,p]
for cent in maxoids.T:
    axes[2].plot(cent[0],cent[1],'ok')

plt.show()
