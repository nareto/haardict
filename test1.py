from twoDdict import *
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
    
plot = False
#learnimg = 'img/flowers_pool.cr2'
#codeimg = 'img/flowers_pool.cr2'
learnimg = 'img/flowers_pool-rescale.npy'
codeimg = 'img/flowers_pool-rescale.npy'
#learnimg = 'img/flowers_pool-small.npy'
#codeimg = 'img/flowers_pool-small.npy'
patch_size = (16,16)
#patch_size = (8,8)
#npatches = None
npatches = 5000
sparsity = 2
#meth = '2ddict'
meth = 'ksvd'
#test_meths = ['ksvd']
clust = '2means'
#clust = 'spectral'
#cluster_epsilon = 3200000
#cluster_epsilon = 3e-4
cluster_epsilon = 3e7
#spectral_dissimilarity = 'haarpsi'
spectral_dissimilarity = 'emd'
#spectral_distance = 'euclidean'
#cluster_epsilon = 10
#learn_transf = 'wavelet'
#learn_transf = 'wavelet_packet'
#learn_transf = '2dpca'
learn_transf = None
tdpcal,tdpcar = 4,4
rec_transf = None
#rec_transf = 'wavelet_packet'
ksvd_cardinality = 11

### LEARNING ###
ksvd_sparsity = sparsity
dictionaries = {}
reconstructed = {}
dwtd = False
if rec_transf is not None:
    dwtd = True
try:
    tic()
except:
    pass
dictionary = learn_dict([learnimg],npatches,patch_size,method=meth,clustering=clust,transform=learn_transf,cluster_epsilon=cluster_epsilon,spectral_dissimilarity=spectral_dissimilarity,ksvddictsize=ksvd_cardinality,ksvdsparsity=ksvd_sparsity,twodpca_l=tdpcal,twodpca_r=tdpcar,dict_with_transformed_data=dwtd)
try:
    print('Learned dictionary in %f seconds' % toc(0))
except:
    pass

### RECONSTRUCT ###
try:
    tic()
except:
    pass
rec = reconstruct(dictionary,codeimg,sparsity,rec_transf)
try:
    print('Reconstructed image in %f seconds' % toc(0))
except:
    pass
rec = rescale(rec,True)
img = np_or_img_to_array(codeimg,patch_size)
hpi = haar_psi(img,rec)[0]
psnrval = psnr(img,rec)
twonorm = np.linalg.norm(img-rec,ord=2)
#fronorm = np.linalg.norm(img-rec,ord='fro')

print('\n\nLearn img: %s\nReconstruction img: %s\nLearning method: %s \n Learning Transform: %s\nClustering method: %s \nCluster epsilon: %f\nSpectral distance: %s\nReconstruction Transform: %s\nDictionary cardinality: %d'
      %(learnimg,codeimg,meth,learn_transf,clust,cluster_epsilon,spectral_distance,rec_transf,dictionary.cardinality))
print(orgmode_table_line(['PSNR','HaarPSI']))
print(orgmode_table_line([psnrval,hpi]))
if plot:
    fig, (ax1, ax2) = plt.subplots(1, 2)#, sharey=True)
    #ax1.imshow(img[34:80,95:137], cmap=plt.cm.gray,interpolation='none')
    #ax2.imshow(rec[34:80,95:137], cmap=plt.cm.gray,interpolation='none')
    ax1.imshow(img, cmap=plt.cm.gray,interpolation='none')
    ax2.imshow(rec, cmap=plt.cm.gray,interpolation='none')
    #ax3.imshow(recclip, cmap=plt.cm.gray,interpolation='none')
    #fig.show()
    plt.show()

#return(dictionaries,rec)

