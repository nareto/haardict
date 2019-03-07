from haardict import *
import numpy as np

np.random.seed(123)

learnimgs = 'img/flowers_pool-rescale.npy'
#learnimgs = 'img/cameraman256.png'
#learnimgs = 'img/boat512.png'
#learnimgs = ['img/cameraman256.png','img/lena512.png','img/barbara512.png','img/peppers256.png']

#codeimg = 'img/landscape2-rescaled.jpg'
codeimg = 'img/flowers_pool-rescale.npy'
#codeimg = 'img/boat512.png'

patch_size = (16,16)
#patch_size = (24,24)
#patch_size = (32,32)
#patch_size = (8,8)
#npatches = None
npatches = 500

#base_save_dir = '/Users/renato/tmp/'
#base_save_dir = '/Users/renato/nextcloud/phd/'
#save_prefix = 'jimg/'+date+'-'+testid

dictionary_cardinality = 384
sparsity = 5
t = Test(learnimgs,npatches,patch_size,noisevar=0,overlapped_patches=False)
t.learn_dict(method='haar-dict', dictsize=dictionary_cardinality, clustering='twomaxoids')
#t.learn_dict(method='haar-dict', dictsize=dictionary_cardinality, clustering='spectral', spectral_similarity='haarpsi')
#t.overlapped_patches = False
t.reconstruct(codeimg,sparsity)
t.dictionary.show_most_used_atoms(t.rec_coefs)
t.print_results()


#ksvd = Test(learnimgs,npatches,patch_size,noisevar=0,overlapped_patches=False)
#ksvd.learn_dict(method='ksvd', dictsize=dictionary_cardinality,ksvdsparsity=sparsity)
##ksvd.overlapped_patches = False
#ksvd.reconstruct(codeimg,sparsity)
#ksvd.print_results()
#ksvd.dictionary.show_most_used_atoms(ksvd.rec_coefs)
#t.show_rec_img()
#ksvd.show_rec_img()
