import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import twoDdict as td

# WE USE THIS SCRIPT TO MAKE VARIOUS PLOTS REGARDING SPECTRAL CLUSTERING
seed = 234
np.random.seed(seed)

patches = td.extract_patches(td.np_or_img_to_array('img/flowers_pool-rescale.npy'))                                          
p500 = [patches[i] for i in np.random.permutation(range(len(patches)))][:500]
p750 = [patches[i] for i in np.random.permutation(range(len(patches)))][:750]
p50 = [patches[i] for i in np.random.permutation(range(len(patches)))][:50]                                                
p150 = [patches[i] for i in np.random.permutation(range(len(patches)))][:150]                                                
#dd = td.learn_dict(['img/flowers_pool-rescale.npy'])                                                                         
scemd750 = td.spectral_clustering(p750,1e-4,'emd')
scemd50 = td.spectral_clustering(p50,1e-4,'emd')
schaar50 = td.spectral_clustering(p50,1e-4,'haarpsi')
schaar750 = td.spectral_clustering(p750,1e-4,'haarpsi')
scfro750 = td.spectral_clustering(p750,1e-4,'frobenius')

def exec(loadpickle=False):
    for simm,beta in zip(['frobenius','haarpsi','emd'],[0.06,1,0.001]):
        sc = td.spectral_clustering(p750,1e-4,simm,simmeasure_beta=beta)
        codename = 'scp750clust-seed:'+str(seed)+'-'+simm+'.pickle'
        if loadpickle:
            sc.load_pickle(codename)
        else:
            sc.compute()
            sc.save_pickle(codename)
        plt.gcf().clear()
        plotshist(sc,'outimg/hist-'+simm+'.png')
        plt.gcf().clear()
        plotegv(sc,'outimg/egv-'+simm+'.png')
        plt.gcf().clear()

def plotshist(clust,savep=None):
    nb = int(len(clust.samples)/5)
    plt.hist(clust.affinity_matrix.data,bins=nb)
    if savep is not None:
        plt.savefig(savep)
    else:
        plt.show()

def plotegv(clust,savep=None):
    egv = clust.egvecs[0][1]
    egv.sort()
    isinleftcluster = egv > threshold_otsu(egv)

    t = np.arange(0,len(egv))
    split = isinleftcluster.argmax()
    print(split)
    leftt = t[:split]
    rightt = t[split:]
    plt.plot(leftt,egv[:split],'r-')
    plt.plot(rightt,egv[split:],'b-')
    #plt.plot(0.2*isinleftcluster,'g')
    if savep is not None:
        plt.savefig(savep)
    else:
        plt.show()




