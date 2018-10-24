#    Copyright 2018 Renato Budinich
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from haardict import *
import numpy as np
import matplotlib.pyplot as plt
import copy

np.random.seed(123)

save = False
#save = True

#figures = True
figures = False
testid = 'fprint'
#testid = 'fp1->fp2'
now = dt.datetime.now()
date = '_'.join(map(str,[now.year,now.month,now.day])) + '-'+'-'.join(map(str,[now.hour,now.minute]))
#base_save_dir = '/Users/renato/tmp/'
base_save_dir = '/Users/renato/nextcloud/phd/'
save_prefix = 'jimg/'+date+'-'+testid
#learnimg = 'img/flowers_pool.cr2'
#codeimg = 'img/flowers_pool.cr2'
#learnimgs = ['img/flowers_pool-rescale.npy']
#learnimgs = ['img/flowers_pool-rescale.npy','img/boat512.png','img/barbara512.png']
#learnimgs = ['img/flowers_pool-rescale.npy', 'img/house256.png','img/cameraman256.png','img/barbara512.png']
learnimgs = ['img/cameraman256.png','img/lena512.png','img/barbara512.png','img/peppers256.png']
#learnimgs = ['img/peppers256.png']
#learnimgs = ['img/cameraman256.png']
#learnimgs = ['img/landscape1-rescaled.jpg']
#learnimgs = ['img/fingerprint512.png','img/fprint1.png','img/fprint2.png']
#learnimgs = ['img/fingerprint512.png','img/fprint1.png']
#learnimgs = ['img/seis0_orig.eps','img/seis2_orig.eps']
#codeimg = 'img/landscape2-rescaled.jpg'
#codeimg = 'img/flowers_pool-rescale.npy'
codeimg = 'img/boat512.png'
#codeimg = 'img/cameraman256.png'
#codeimg = 'img/peppers256.png'
#codeimg = 'img/fprint3.bmp'
#codeimg = 'img/fprint2.png'
#codeimg = 'img/seis3.eps'
#codeimg = 'img/rocks_lake-rescaled.npy'
#codeimg = 'img/barbara.jpg'
#learnimg = 'img/flowers_pool-small.npy'
#codeimg = 'img/flowers_pool-small.npy'
#patch_size = (16,16)
#patch_size = (24,24)
#patch_size = (32,32)
patch_size = (8,8)
npatches = None
#npatches = 300
sparsity = 5
#meth = '2ddict'
meth = 'ksvd'
#meth = 'warmstart'
#test_meths = ['ksvd']
clust = 'twomeans'
#clust = 'spectral'
#clust = 'random'

noise = 0

dictsize = 85
#dictsize = None
cluster_epsilon = None
#cluster_epsilon = 3e-4 #for emd spectral on 8x8 patches -> 47 card. for haarpsi -> 83
#cluster_epsilon = 5e-6
#cluster_epsilon = 52e-4
#cluster_epsilon = 135e-3
#cluster_epsilon = 42.5
#cluster_epsilon = 8e-2
#cluster_epsilon = 1.5e-2
#cluster_epsilon = 0.3e-1
#cluster_epsilon = 1e-4 #-> 71 for spectral haarpsi on 8x8, 47 for emd
#cluster_epsilon = 2e-5#-> for emd gives 44
#cluster_epsilon = 1500
b = 64 #bits used for encoding one float

#SPECTRAL CLUSTERING

#spectral_similarity = 'haarpsi'
#spectral_similarity = 'emd'
spectral_similarity = 'frobenius'
affinity_matrix_threshold = 0.5
simmeasure_beta = 0.5 #only for Frobenius and EMD similarity measures

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
mc = None
compute_mutual_coherence = False


### LEARNING ###
ksvd_sparsity = sparsity
dictionaries = {}
reconstructed = {}
dwtd = False
#if rec_transf is not None:
#    dwtd = True

### PLOTTING
#show_sc_egvs = True
show_sc_egvs = False

def main():
    img = np_or_img_to_array(codeimg,patch_size)

    ### LEARN DICTIONARY ###
    dictionary,learn_time = learn_dict(learnimgs,npatches,patch_size,noisevar=noise,method=meth,dictsize=dictsize,clustering=clust,transform=learn_transf,cluster_epsilon=cluster_epsilon,spectral_similarity=spectral_similarity,simmeasure_beta=simmeasure_beta,affinity_matrix_threshold=affinity_matrix_threshold,ksvdsparsity=ksvd_sparsity,twodpca_l=tdpcal,twodpca_r=tdpcar,dict_with_transformed_data=dwtd,wavelet=wav,wav_lev=wavlev,dicttype='haar')

    dictionary.useksvdencoding = False

    ### RECONSTRUCT ###
    rec,coefs,rec_time,noisy_img = reconstruct(dictionary,codeimg,patch_size,sparsity,noisevar=noise)
    print_test_parameters(dictionary,learn_time,rec_time)
    storage = storage_cost(dictionary,coefs, sparsity, bits = b)
    print_rec_results(dictionary,rec,img,coefs,storage,learn_time,noisy_img=noisy_img)
    if meth == '2ddict':
        dictionary2 = copy.deepcopy(dictionary)
        dictionary2.set_dicttype('centroids')
        rec2,coefs2,time,noisy_img2 = reconstruct(dictionary2,codeimg,patch_size,sparsity,noisevar=noise)
        storage2 = storage_cost(dictionary2,coefs2, sparsity, bits = 64)
        print_rec_results(dictionary2,img,rec2,coefs2,storage2,learn_time,noisy_img=None,firstorgline=False)
    if meth == '2ddict':
        tag = dictionary.dicttype + ':'
        tag2 = dictionary2.dicttype + ':'
    else:
        tag = ''
    if figures:
        print_and_save_figures(dictionary,img,rec,coefs,tag,noisy_img=noisy_img,simhist=True if (meth == '2ddict' and clust == 'spectral') else False)
        if meth == '2ddict':
            print_and_save_figures(dictionary2,img,rec2,coefs2,tag2,simhist=False)
        if show_sc_egvs:
            dictionary.clustering.plotegvecs()

def print_test_parameters(dictionary,learn_time,rec_time):
    desc_string = '\n'+10*'-'+'Test results -- '+str(dt.datetime.now())+10*'-'
    desc_string += '\nLearn imgs: %s\nReconstruction img: %s\nPatch size: %s\nN. of patches: %d\nLearning method: %s\nCoding sparsity: %d\nDictionary learning time: %4.2f\nReconstruction time: %4.2f\nTotal time: %4.2f' % \
                   (learnimgs,codeimg,patch_size,len(dictionary.patches),meth,sparsity,learn_time,rec_time,learn_time+rec_time)
    if learn_transf is not None:
        desc_string += '\nLearning transform: %s' % (learn_transf)
        if learn_transf in ['wavelet','wavelet_packet']:
            desc_string += ' - %s' % wav
        if learn_transf is '2dpca':
            desc_string += ' - l = %d, r = %d' % (tdpcal,tdpcar)
    if noise is not 0:
        desc_string += '\nNoise variance: %f' % noise
    if meth == '2ddict':
        if dictionary.visit == 'fifo':
            desc_string += '\nClustering method: %s \nCluster epsilon: %f\nTree depth: %d\nTree sparsity: %f' % (clust,cluster_epsilon,dictionary.tree_depth,dictionary.tree_sparsity)
        else:
            desc_string += '\nClustering method: %s \nTree depth: %d\nTree sparsity: %f' % (clust,dictionary.tree_depth,dictionary.tree_sparsity)
        if clust == 'spectral':
            desc_string += '\nSpectral similarity measure: %s\nAffinity matrix sparsity: %f' % (spectral_similarity,dictionary.clustering.affinity_matrix_nonzero_perc)
    print(desc_string)

    
def print_rec_results(dictionary,img,rec,coefs,storage_cost,learn_time,noisy_img=None,firstorgline=True):
    if noisy_img is not None:
        n_hpi = haar_psi(255*img,255*noisy_img)[0]
        n_psnrval = psnr(img,noisy_img)
    else:
        n_hpi = '-'
        n_psnrval = '-'
    rec_hpi = haar_psi(255*img,255*rec)[0]
    rec_psnrval = psnr(img,rec)
    ent = entropy(atoms_prob(coefs))
    qindex = b*img.size*rec_hpi/storage_cost
    if compute_mutual_coherence:
        dictionary.mutual_coherence(False)
        mc = dictionary.max_cor
        argmc = dictionary.argmax_cor
        mcstring = str(mc)+' (atoms %d and %d)'%(argmc[0],argmc[1])
    else:
        mcstring = '-'
    if meth == '2ddict':
        if dictionary.visit == 'fifo':
            meth_string = '-'.join((meth,clust,dictionary.dicttype,('%.2e' % cluster_epsilon)))
        else:
            meth_string = '-'.join((meth,clust,dictionary.dicttype))
    else:
        meth_string = meth
    if firstorgline:
        print('\n')
        print(orgmode_table_line(['Method','K', 'Patch Size', 'Learn Time', 'Noisy PSNR','Noisy HaarPSI','Reconstructed PSNR','Reconstructed HaarPSI','Storage Cost', 'Q index']))
        
    print(orgmode_table_line([meth_string,dictionary.dictsize,dictionary.patch_size,learn_time,n_psnrval,n_hpi,rec_psnrval,rec_hpi,storage_cost,qindex]))


def print_and_save_figures(dictionary,img,rec,coefs,tag,noisy_img=None,simhist=False):
    savename = save_prefix + '-%s-%s' % (meth,tag.rstrip(':'))
    savename += '-%dx%d' % patch_size
    if meth == '2ddict':
        if clust == 'twomeans':
            savename += '-2means'
        elif clust == 'spectral':
            savename += '-spec_%s' % spectral_similarity

    if simhist:
        nb = int(len(dictionary.patches)/5)
        plt.hist(dictionary.clustering.affinity_matrix.data,bins=nb)
        simhistpath = savename+'-similarities.png'
        if save:
            plt.savefig(base_save_dir+simhistpath)
        plt.close()
        orgmode_str= '**** similarity histograms\n[[file:%s]]\n' % (simhistpath)
        print(orgmode_str)
    if noisy_img is not None:
        nimg = savename+'-noisy_image.png'
        plt.imshow(noisy_img)
        if save:
            plt.savefig(base_save_dir+nimg)
        plt.close()
        orgmode_str = '\n**** noisy image\n[[file:%s]]\n' % nimg
        print(orgmode_str)
    recimg = savename+'-reconstructed_image.png'
    plt.imshow(rec)
    if save:
        plt.savefig(base_save_dir+recimg)
    plt.close()
    orgmode_str = '\n**** %s reconstructed image\n[[file:%s]]\n' % (tag,recimg)
    print(orgmode_str)
    dictelements = savename+'-showdict.png'
    orgmode_str = '**** %s showdict\n[[file:%s]]\n' % (tag,dictelements)
    if save:
        dictionary.show_dict_patches(savefile=base_save_dir+dictelements)
    print(orgmode_str)
    plt.close()

    mostused = min(dictionary.dictsize,50)
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

def orgmode_table_line(strings_or_n):
    out ='| ' + ' | '.join([str(s) for s in strings_or_n]) + ' |'
    return(out)

            
if __name__ == '__main__':
    main()
