from twoDdict import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime as dt

LAST_TIC = dt.datetime.now()
def tic():
    global LAST_TIC
    LAST_TIC = dt.datetime.now()

def toc(printstr=True):
    global LAST_TIC
    dtobj = dt.datetime.now() - LAST_TIC
    ret = dtobj.total_seconds()
    if printstr:
        print('%f seconds elapsed' % ret)
    return(ret)

np.random.seed(123)

#save = False
save = True

figures = True
#testid = 'flowers-transf'
testid = 'fp1->fp2'
now = dt.datetime.now()
date = '_'.join(map(str,[now.year,now.month,now.day])) + '-'+'-'.join(map(str,[now.hour,now.minute]))
#base_save_dir = '/Users/renato/tmp/'
base_save_dir = '/Users/renato/nextcloud/phd/'
save_prefix = 'jimg/'+date+'-'+testid
#learnimg = 'img/flowers_pool.cr2'
#codeimg = 'img/flowers_pool.cr2'
#learnimg = 'img/flowers_pool-rescale.npy'
#codeimg = 'img/flowers_pool-rescale.npy'
learnimg = 'img/fprint1.png'
codeimg = 'img/fprint2.png'
#codeimg = 'img/rocks_lake-rescaled.npy'
#codeimg = 'img/barbara.jpg'
#learnimg = 'img/flowers_pool-small.npy'
#codeimg = 'img/flowers_pool-small.npy'
#patch_size = (16,16)
#patch_size = (24,24)
#patch_size = (32,32)
patch_size = (8,8)
npatches = None
#npatches = 500
sparsity = 2
#meth = '2ddict'
meth = 'ksvd'
#test_meths = ['ksvd']
clust = '2means'
#clust = 'spectral'

ksvd_cardinality = 

#cluster_epsilon = 3e-4 #for emd spectral on 8x8 patches -> 47 card. for haarpsi -> 83
cluster_epsilon = 1.5e1
#cluster_epsilon = 1e-4 #-> 71 for spectral haarpsi on 8x8, 47 for emd
#cluster_epsilon = 2e-5#-> for emd gives 44
#cluster_epsilon = 1500

#SPECTRAL CLUSTERING
spectral_similarity = 'haarpsi'
#spectral_similarity = 'emd'
#spectral_similarity = 'frobenius'
affinity_matrix_threshold = 0.5
simmeasure_beta = 0.06 #only for Frobenius and EMD similarity measures

#TRANSFORMS
#learn_transf = 'wavelet'
#learn_transf = 'wavelet'
#learn_transf = 'wavelet_packet'
#learn_transf = '2dpca'
learn_transf = None
#wav = 'db2'
wav = 'haar'
wavlev = 1
tdpcal,tdpcar = 4,4
rec_transf = None
#rec_transf = 'wavelet'
#rec_transf = 'wavelet_packet'
#rec_transf = '2dpca'
mc = None
compute_mutual_coherence = True


### LEARNING ###
ksvd_sparsity = sparsity
dictionaries = {}
reconstructed = {}
dwtd = False
if rec_transf is not None:
    dwtd = True

### PLOTTING
#show_sc_egvs = True
show_sc_egvs = False

def main():
    img = np_or_img_to_array(codeimg,patch_size)
    tic()
    dictionary = learn_dict([learnimg],npatches,patch_size,method=meth,clustering=clust,transform=learn_transf,cluster_epsilon=cluster_epsilon,spectral_similarity=spectral_similarity,simmeasure_beta=simmeasure_beta,affinity_matrix_threshold=affinity_matrix_threshold,ksvddictsize=ksvd_cardinality,ksvdsparsity=ksvd_sparsity,twodpca_l=tdpcal,twodpca_r=tdpcar,dict_with_transformed_data=dwtd,wavelet=wav,wav_lev=wavlev,dicttype='haar')
    elapsed_time = toc(0)

    print('Learned dictionary in %f seconds' % elapsed_time)
    print_test_parameters(dictionary,elapsed_time)
    ### RECONSTRUCT ###
    #tic()
    rec,coefs = reconstruct(dictionary,codeimg,patch_size,sparsity,rec_transf,twodpca_l=tdpcal,twodpca_r=tdpcar,wavelet=wav,wav_lev=wavlev)
    #print('Reconstructed image in %f seconds' % toc(0))
    #rec = rescale(rec,True)
    print_rec_results(dictionary,rec,img,coefs)
    if meth == '2ddict':
        dictionary2 = copy.deepcopy(dictionary)
        dictionary2.set_dicttype('centroids')    
        #tic()
        rec2,coefs2 = reconstruct(dictionary2,codeimg,patch_size,sparsity,rec_transf,wavelet=wav,wav_lev=wavlev)
        #print('Reconstructed image in %f seconds' % toc(0))
        #rec = rescale(rec,True)
        print_rec_results(dictionary2,img,rec2,coefs2,False)
    if meth == '2ddict':
        tag = dictionary.dicttype + ':'
        tag2 = dictionary2.dicttype + ':'
    else:
        tag = ''
    if figures:
        print_and_save_figures(dictionary,img,rec,coefs,tag)
        if meth == '2ddict':
            print_and_save_figures(dictionary2,img,rec2,coefs2,tag2)
        if show_sc_egvs:
            dictionary.clustering.plotegvecs()

def print_test_parameters(dictionary,elapsed_time):
    desc_string = '\n'+10*'-'+'Test results -- '+str(dt.datetime.now())+10*'-'
    desc_string += '\nLearn img: %s\nReconstruction img: %s\nPatch size: %s\nN. of patches: %d\nLearning method: %s\nCoding sparsity: %d\nElapsed time: %4.2f' % \
                   (learnimg,codeimg,patch_size,len(dictionary.patches),meth,sparsity,elapsed_time)
    #if mc is not None:
    #    desc_string += '\nMutual coherence: %f (from atoms %d and %d)' % (mc,argmc[0],argmc[1])
    if learn_transf is not None:
        desc_string += '\nLearning transform: %s' % (learn_transf)
        if learn_transf is not '2dpca':
            desc_string += ' - %s' % wav
    if meth == '2ddict':
        desc_string += '\nClustering method: %s \nCluster epsilon: %f\nTree depth: %d\nTree sparsity: %f' % (clust,cluster_epsilon,dictionary.clustering.tree_depth,dictionary.clustering.tree_sparsity)
        if clust == 'spectral':
            desc_string += '\nSpectral similarity measure: %s\nAffinity matrix sparsity: %f' % (spectral_similarity,dictionary.clustering.affinity_matrix_nonzero_perc)
    if rec_transf is not None:
        desc_string += '\nReconstruction transform: %s' % (rec_transf)

    print(desc_string)

    
def print_rec_results(dictionary,img,rec,coefs,firstorgline=True):
    hpi = haar_psi(255*img,255*rec)[0]
    psnrval = psnr(img,rec)
    ent = entropy(atoms_prob(coefs))
    if compute_mutual_coherence:
        dictionary.mutual_coherence(False)
        mc = dictionary.max_cor
        argmc = dictionary.argmax_cor
        mcstring = str(mc)+' (atoms %d and %d)'%(argmc[0],argmc[1])
    else:
        mcstring = '-'
    #reconstructed[dict_string] = 
    #twonorm = np.linalg.norm(img-rec,ord=2)
    #fronorm = np.linalg.norm(img-rec,ord='fro')

    if meth == '2ddict':
        meth_string = '-'.join((meth,clust,dictionary.dicttype,('%.2e' % cluster_epsilon)))
    else:
        meth_string = meth
    if firstorgline:
        print('\n')
        print(orgmode_table_line(['Method','K','Mutual Coeherence', 'Entropy','PSNR','HaarPSI']))
    print(orgmode_table_line([meth_string,dictionary.cardinality,mcstring,ent,psnrval,hpi]))


def print_and_save_figures(dictionary,img,rec,coefs,tag):
    savename = save_prefix + '-%s-%s' % (meth,tag.rstrip(':'))
    savename += '-%dx%d' % patch_size
    if meth == '2ddict':
        if clust == '2means':
            savename += '-2means'
        else:
            savename += '-spec_%s' % spectral_similarity

    recimg = savename+'-reconstructed_image.png'
    plt.imshow(rec)
    if save:
        plt.savefig(base_save_dir+recimg)
    plt.close()
    orgmode_str = '\n**** %s reconstructed image\n[[file:%s]]\n' % (tag,recimg)
    print(orgmode_str)
    if meth == '2ddict' and clust == 'spectral':
        nb = int(len(clust.samples)/5)
        plt.hist(dictionary.clustering.affinity_matrix.data,bins=nb)
        simhist = savename+'-similarities.png'
        if save:
            plt.savefig(base_save_dir+simhist)
        plt.close()
        orgmode_str= '**** %s similarity histograms\n[[file:%s]]\n' % (tag,simhist)
        print(orgmode_str)
    dictelements = savename+'-showdict.png'
    orgmode_str = '**** %s showdict\n[[file:%s]]\n' % (tag,dictelements)
    if save:
        dictionary.show_dict_patches(savefile=base_save_dir+dictelements)
    print(orgmode_str)
    plt.close()

    mostused = min(dictionary.cardinality,50)
    mostusedatoms = savename+'-mostused.png'
    orgmode_str = '**** %s %d most used atoms\n[[file:%s]]' % (tag,mostused,mostusedatoms)
    if save:
        dictionary.show_most_used_atoms(coefs,mostused,savefile=base_save_dir+mostusedatoms)
    print(orgmode_str)
    plt.close()

    atomsprob1 = savename+'-atoms_prob.png'
    orgmode_str = '**** %s atoms prob\n[[file:%s]]\n' % (tag, atomsprob1)
    atomsprob2 = savename+'-atoms_prob-sorted.png'
    orgmode_str += '[[file:%s]]'% (atomsprob2)
    pr = atoms_prob(coefs)
    plt.plot(pr)
    if save:
        plt.savefig(base_save_dir+atomsprob1)
    plt.close()
    pr.sort()
    pr = pr[::-1]
    plt.plot(pr)
    if save:
        plt.savefig(base_save_dir+atomsprob2)
    plt.close()
    print(orgmode_str)

    if npatches is not None and npatches < 100:
        orgmode_str= '**** %s min sim histograms\n' % tag
        print(orgmode_str)
        for sim in ['frobenius','haarpsi','emd']:
            fig = plt.figure()
            dictionary.max_similarities(sim,False)
            plt.hist(dictionary.max_sim)
            simhist = savename+'-maxsim_%s'%(sim)+'.png'
            orgmode_str = '[[file:%s]]' % simhist
            if save:
                fig.savefig(base_save_dir+simhist)
            fig.clear()
            plt.close()
            print(orgmode_str)

if __name__ == '__main__':
    main()
