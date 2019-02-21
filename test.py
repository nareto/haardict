from haardict import *
import numpy as np

np.random.seed(123)

#learnimgs = 'img/flowers_pool-rescale.npy'
learnimgs = 'img/cameraman256.png'
#learnimgs = ['img/cameraman256.png','img/lena512.png','img/barbara512.png','img/peppers256.png']

#codeimg = 'img/landscape2-rescaled.jpg'
#codeimg = 'img/flowers_pool-rescale.npy'
codeimg = 'img/boat512.png'

patch_size = (16,16)
#patch_size = (24,24)
#patch_size = (32,32)
#patch_size = (8,8)
#npatches = None
npatches = 110

#base_save_dir = '/Users/renato/tmp/'
#base_save_dir = '/Users/renato/nextcloud/phd/'
#save_prefix = 'jimg/'+date+'-'+testid


t = Test(learnimgs,npatches,patch_size,noisevar=0.3,overlapped_patches=False)
t.learn_dict(method='haar-dict', dictsize=6, clustering='spectral')
#t.overlapped_patches = False
t.reconstruct(codeimg,5)
#t.show_rec_img()
t.print_results()
#
#t.learn_dict(method='haar-dict', dictsize=6, clustering='twomeans')
#t.reconstruct(codeimg,5)
##t.show_rec_img()
#t.print_results()
#t.print_and_save_orgmode('asdf/',simhist=False,saveimgs=True)

#ksvd = Test(learnimgs,npatches,patch_size)
#ksvd.learn_dict(method='ksvd', dictsize=85)
#ksvd.reconstruct(codeimg,5)
##ksvd.show_rec_img()
#ksvd.print_results()
