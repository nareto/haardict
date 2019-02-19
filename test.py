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


hd = Test(learnimgs,npatches,patch_size,noisevar=0,overlapped_patches=True)
hd.learn_dict(method='haar-dict', dictsize=6, clustering='twomaxoids')
#hd.overlapped_patches = False
hd.reconstruct(codeimg,5)
#hd.show_rec_img()
hd.print_results()
#
#hd.learn_dict(method='haar-dict', dictsize=6, clustering='twomeans')
#hd.reconstruct(codeimg,5)
##hd.show_rec_img()
#hd.print_results()
hd.print_and_save_orgmode('asdf/',simhist=False,saveimgs=True)

#ksvd = Test(learnimgs,npatches,patch_size)
#ksvd.learn_dict(method='ksvd', dictsize=85)
#ksvd.reconstruct(codeimg,5)
##ksvd.show_rec_img()
#ksvd.print_results()
