from haardict import *
import numpy as np

np.random.seed(123)

#learnimgs = 'img/flowers_pool-rescale.npy'
#learnimgs = 'img/cameraman256.png'
learnimgs = 'img/boat512.png'
#learnimgs = ['img/cameraman256.png','img/lena512.png','img/barbara512.png','img/peppers256.png']

#codeimg = 'img/landscape2-rescaled.jpg'
#codeimg = 'img/flowers_pool-rescale.npy'
codeimg = 'img/boat512.png'

patch_size = (16,16)
#patch_size = (24,24)
#patch_size = (32,32)
#patch_size = (8,8)
#npatches = None
npatches = 4000

#base_save_dir = '/Users/renato/tmp/'
#base_save_dir = '/Users/renato/nextcloud/phd/'
#save_prefix = 'jimg/'+date+'-'+testid

dictionary_cardinality = 40
sparsity = 3
t = Test(learnimgs,npatches,patch_size,noisevar=0,overlapped_patches=False)
t.learn_dict(method='haar-dict', dictsize=dictionary_cardinality, clustering='twomeans')
#t.overlapped_patches = False
t.reconstruct(codeimg,sparsity)
t.print_results()


ksvd = Test(learnimgs,npatches,patch_size,noisevar=0,overlapped_patches=False)
ksvd.learn_dict(method='ksvd', dictsize=dictionary_cardinality,ksvdsparsity=sparsity)
#ksvd.overlapped_patches = False
ksvd.reconstruct(codeimg,sparsity)
ksvd.print_results()

#t.show_rec_img()
ksvd.show_rec_img()
