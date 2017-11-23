from twoDdict import *
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import copy

np.random.seed(123)

save = True

figures = True
testid = 'flowers500-8x8'
now = dt.datetime.now()
date = '_'.join(map(str,[now.year,now.month,now.day])) + '-'+'-'.join(map(str,[now.hour,now.minute]))
save_prefix = '/Users/renato/nextcloud/phd/jimg/'+date+'-'+testid
#learnimg = 'img/flowers_pool.cr2'
#codeimg = 'img/flowers_pool.cr2'
learnimg = 'img/flowers_pool-rescale.npy'
codeimg = 'img/flowers_pool-rescale.npy'
#codeimg = 'img/rocks_lake-rescaled.npy'
#codeimg = 'img/barbara.jpg'
#learnimg = 'img/flowers_pool-small.npy'
#codeimg = 'img/flowers_pool-small.npy'
#patch_size = (16,16)
#patch_size = (24,24)
#patch_size = (32,32)
patch_size = (8,8)
#npatches = None
npatches = 500
sparsity = 2
meth = '2ddict'
#meth = 'ksvd'
#test_meths = ['ksvd']
#clust = '2means'
clust = 'spectral'
#cluster_epsilon = 3e-4 #for emd spectral on 8x8 patches -> 47 card. for haarpsi -> 83
#cluster_epsilon = 1e-4
cluster_epsilon = 1e-4
spectral_dissimilarity = 'haarpsi'
#spectral_dissimilarity = 'emd'
#spectral_dissimilarity = 'euclidean'
#cluster_epsilon = 10
#learn_transf = 'wavelet'
#wav = 'db2'
wav = 'haar'
wavlev = 1
#learn_transf = 'wavelet_packet'
#learn_transf = '2dpca'
learn_transf = None
tdpcal,tdpcar = 4,4
rec_transf = None
#rec_transf = 'wavelet'
#rec_transf = 'wavelet_packet'
mc = None
compute_mutual_coherence = True
ksvd_cardinality = 427


### LEARNING ###
ksvd_sparsity = sparsity
dictionaries = {}
reconstructed = {}
dwtd = False
if rec_transf is not None:
    dwtd = True

def main():
    img = np_or_img_to_array(codeimg,patch_size)
    tic()
    dictionary = learn_dict([learnimg],npatches,patch_size,method=meth,clustering=clust,transform=learn_transf,cluster_epsilon=cluster_epsilon,spectral_dissimilarity=spectral_dissimilarity,ksvddictsize=ksvd_cardinality,ksvdsparsity=ksvd_sparsity,twodpca_l=tdpcal,twodpca_r=tdpcar,dict_with_transformed_data=dwtd,wavelet=wav,wav_lev=wavlev,dicttype='haar')
    elapsed_time = toc(0)

    print('Learned dictionary in %f seconds' % elapsed_time)
    print_test_parameters(dictionary,elapsed_time)
    ### RECONSTRUCT ###
    #tic()
    rec,coefs = reconstruct(dictionary,codeimg,sparsity,rec_transf,wavelet=wav,wav_lev=wavlev)
    #print('Reconstructed image in %f seconds' % toc(0))
    #rec = rescale(rec,True)
    print_rec_results(dictionary,rec,img,coefs)
    if meth == '2ddict':
        dictionary2 = copy.deepcopy(dictionary)
        dictionary2.set_dicttype('centroids')    
        #tic()
        rec2,coefs2 = reconstruct(dictionary2,codeimg,sparsity,rec_transf,wavelet=wav,wav_lev=wavlev)
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
            desc_string += '\nSpectral dissimilarity measure: %s\nAffinity matrix sparsity: %f' % (spectral_dissimilarity,dictionary.clustering.affinity_matrix_nonzero_perc)
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
    basesavepath = save_prefix + '-%s-%s' % (meth,tag.rstrip(':'))
    basesavepath += '-%dx%d' % patch_size
    if meth == '2ddict':
        if clust == '2means':
            basesavepath += '-2means'
        else:
            basesavepath += '-spec_%s' % spectral_dissimilarity

    savepath = basesavepath+'-reconstructed_image.png'
    plt.imshow(rec)
    if save:
        plt.savefig(savepath)
    plt.close()
    orgmode_str = '\n**** %s reconstructed image\n[[file:%s]]\n' % (tag,savepath)
    print(orgmode_str)
    if meth == '2ddict' and clust == 'spectral':
        plt.hist(dictionary.clustering.affinity_matrix.data)
        savepath = basesavepath+'-dissimilarities.png'
        if save:
            plt.savefig(savepath)
        plt.close()
        orgmode_str= '**** %s dissimilarity histograms\n[[file:%s]]\n' % (tag,savepath)
        print(orgmode_str)
    savepath = basesavepath+'-showdict.png'
    orgmode_str = '**** %s showdict\n[[file:%s]]\n' % (tag,savepath)
    if save:
        dictionary.show_dict_patches(savefile=savepath)
    print(orgmode_str)
    plt.close()

    mostused = min(dictionary.cardinality,50)
    mostused = int(np.sqrt(mostused))**2
    savepath = basesavepath+'-mostused.png'
    orgmode_str = '**** %s %d most used atoms\n[[file:%s]]' % (tag,mostused,savepath)
    if save:
        dictionary.show_most_used_atoms(coefs,mostused,savefile=savepath)
    plt.close()

    savepath1 = basesavepath+'-atoms_prob.png'
    orgmode_str = '**** %s atoms prob\n[[file:%s]]\n' % (tag, savepath1)
    savepath2 = basesavepath+'-atoms_prob-sorted.png'
    orgmode_str += '[[file:%s]]'% (savepath2)
    pr = atoms_prob(coefs)
    plt.plot(pr)
    if save:
        plt.savefig(savepath1)
    plt.close()
    pr.sort()
    pr = pr[::-1]
    plt.plot(pr)
    if save:
        plt.savefig(savepath2)
    plt.close()
    print(orgmode_str)

    if npatches is not None and npatches < 100:
        orgmode_str= '**** %s min diss histograms\n' % tag
        print(orgmode_str)
        for diss in ['euclidean','haarpsi','emd']:
            fig = plt.figure()
            dictionary.min_dissimilarities(diss,False)
            plt.hist(dictionary.min_diss)
            savepath = basesavepath+'-mindiss_%s'%(diss)+'.png'
            orgmode_str = '[[file:%s]]' % savepath
            if save:
                fig.savefig(savepath)
            fig.clear()
            plt.close()
            print(orgmode_str)

if __name__ == '__main__':
    main()
