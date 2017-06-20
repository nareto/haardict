import ipdb
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as sslinalg
import skimage.io
import twoDdict

scatter_plot = False
plot = True
npoints = 400
n,m = (8,8)
k = 2
veca = np.random.uniform(size=n*m).reshape(n,m)
vecb = np.random.uniform(size=n*m).reshape(n,m)

var1 = 1
var2 = 20

patches = []
for i in range(npoints):
    mat = np.random.normal(0,var1)*veca + np.random.normal(0,var2)*vecb
    patches.append(twoDdict.Patch(mat))

pca = twoDdict.pca(patches,k)
pca.compute_pca()

for p in patches:
    p.compute_feature_vector(pca.eigenvectors)

fvectors = np.vstack([x.feature_vector for x in patches])

e1 = np.linalg.norm(veca.flatten()-pca.eigenvectors[:,0])
a1 = np.dot(veca.flatten().transpose(),pca.eigenvectors[:,0])
e2 = np.linalg.norm(vecb.flatten()-pca.eigenvectors[:,1])
a2 = np.dot(vecb.flatten().transpose(),pca.eigenvectors[:,1])
print("%f - %f \n %f - %f" % (e1,a1,e2,a2))

if scatter_plot:
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(fvectors[:,0],fvectors[:,1])

    fig.show()

if plot:
    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(veca/np.linalg.norm(veca),interpolation='nearest')
    ax[1,0].imshow(vecb/np.linalg.norm(vecb),interpolation='nearest')
    eig1 = pca.eigenvectors[:,0].reshape(n,m)
    eig2 = pca.eigenvectors[:,1].reshape(n,m)
    ax[0,1].imshow(eig1/np.linalg.norm(eig1),interpolation='nearest')
    ax[1,1].imshow(eig2/np.linalg.norm(eig2),interpolation='nearest')
    fig.show()
