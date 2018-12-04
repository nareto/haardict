import numpy as np
import matplotlib.pyplot as plt
from kmaxoids import KMaxoids
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


#X, y = make_blobs(n_samples=1000, centers=[(0,2),(20,2),(-5,-6)], n_features=2, random_state=0)
# print(X.shape)
# print(y)
# X, y = make_blobs(n_samples=[3, 3, 4], centers=None, n_features=2, random_state=0)
# print(X.shape)
# print(y)

#make_blobs(n_samples=100, n_features=2, centers=None, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)

def make_gauss_blob(center,pca1,var1=2,var2=0.5,npoints=100):
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
#plt.scatter(b1[:,0],b1[:,1],color='red')
#plt.scatter(b2[:,0],b2[:,1],color='blue')
#plt.scatter(b3[:,0],b3[:,1],color='green')

Y = np.vstack([b1,b2,b3])
#each row is a data point

fig,axes = plt.subplots(1,2)
colors = ['red','green','blue']

#KMeans
kmeans = KMeans(n_clusters=3).fit(Y)
for p,coord in enumerate(Y):
    axes[0].scatter(coord[0],coord[1],color=colors[kmeans.labels_[p]])
for p,c in enumerate(colors):
    cent = kmeans.cluster_centers_[p]
    axes[0].plot(cent[0],cent[1],'ok')


#KMaxoids
kmaxoids = KMaxoids(Y.transpose(),3)
maxoids, clusters = kmaxoids.run()

for idx,c in enumerate(clusters):
    col = colors[idx]
    for data_point_idx in c:
        p = Y[data_point_idx,:]
        axes[1].scatter(p[0],p[1],color=col)
    repr = maxoids[:,idx]
    axes[1].plot(repr[0],repr[1],'ok')
#plt.scatter(X[:,0],X[:,1])
plt.show()
