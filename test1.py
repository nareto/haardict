from twoDdict import *
import numpy as np
import matplotlib.pyplot as plt

plot = False
#learnimg = 'img/flowers_pool-rescale.npy'
#codeimg = 'img/flowers_pool-rescale.npy'
learnimg = 'img/flowers_pool-small.npy'
codeimg = 'img/flowers_pool-small.npy'
sparsity = 2
meth = '2ddict'
#meth = 'ksvd'
#test_meths = ['ksvd']
clust = '2means'
#clust = 'spectral'
#cluster_epsilon = 1.e3
cluster_epsilon = 2
#learn_transf = 'wavelet'
#learn_transf = 'wavelet_packet'
#learn_transf = '2dpca'
learn_transf = None
rec_transf = None
#rec_transf = 'wavelet_packet'
ksvd_cardinality = 583

### LEARNING ###
ksvd_sparsity = sparsity
dictionaries = {}
reconstructed = {}
dwtd = False
if rec_transf is not None:
    dwtd = True
dictionary = learn_dict([learnimg],method=meth,clustering=clust,transform=learn_transf,cluster_epsilon=cluster_epsilon,ksvddictsize=ksvd_cardinality,ksvdsparsity=ksvd_sparsity,dict_with_transformed_data=dwtd)

### RECONSTRUCT ###
rec = reconstruct(dictionary,codeimg,sparsity,rec_transf)
img = np.load(codeimg)
hpi = HaarPSI(img,rec)
psnrval = psnr(img,rec)
twonorm = np.linalg.norm(img-rec,ord=2)
fronorm = np.linalg.norm(img-rec,ord='fro')

print('\n\nLearn img: %s\nReconstruction img: %s\nLearning method: %s \n Learning Transform: %s\nClustering method: %s \nCluster epsilon: %f\nReconstruction Transform: %s\nDictionary cardinality: %d'
      %(learnimg,codeimg,meth,learn_transf,clust,cluster_epsilon,rec_transf,dictionary.cardinality))
print(orgmode_table_line(['Frobenius norm','PSNR','HaarPSI']))
print(orgmode_table_line([fronorm,psnrval,hpi]))
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
