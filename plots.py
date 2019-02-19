from haardict import *
import numpy as np
import sys
import ipdb

np.random.seed(123)

#learnimgs = 'img/flowers_pool-rescale.npy'
learnimgs = 'img/boat512.png'
#learnimgs = ['img/cameraman256.png','img/lena512.png','img/barbara512.png','img/peppers256.png']

#codeimg = 'img/landscape2-rescaled.jpg'
#codeimg = 'img/flowers_pool-rescale.npy'
codeimg = 'img/cameraman256.png'

#patch_sizes_npatches_dictsize = [((8,8),15000),((12,12),10000),((16,16),8000),((24,24),4000),((32,32),1500)]
patch_sizes_npatches = [((8,8),150),((12,12),100)]
#for i,psize,np in enumerate(patch_sizes_npatches):
def run_and_save_pickle(fpath):
    tests = Saveable()
    tests.testlist = []
    for psize,np in patch_sizes_npatches:
        pdim = psize[0]*psize[1]
        dsize = int(pdim*1.5) #dictionary 50% bigger than dimension
        spars = int(pdim*0.05) #sparsity 5% of dimension
        #for meth in ['haar-dict','centroids-dict','ksvd']:
        for meth in ['haar-dict','centroids-dict']:
            if meth == 'ksvd':
                cur_test = Test(learnimgs,np,psize,noisevar=0,overlapped_patches=True)
                cur_test.debug = True
                cur_test.learn_dict(method=meth, dictsize=dsize)
                cur_test.reconstruct(codeimg,spars)
                tests.testlist.append(cur_test)
                tests.save_pickle(fpath)
            else:
                for clust in ['twomeans','twomaxoids']:
                    cur_test = Test(learnimgs,np,psize,noisevar=0,overlapped_patches=True)
                    cur_test.debug = True
                    cur_test.learn_dict(method=meth, dictsize=dsize, clustering='twomaxoids')
                    cur_test.reconstruct(codeimg,spars)
                    tests.testlist.append(cur_test)
                    tests.save_pickle(fpath)

def print_test_results(pickled_tests):
    tests = Saveable()
    tests.load_pickle(pickled_tests)
    for t in tests.testlist:
        t.print_results()
        
if __name__ == '__main__':
    run_and_save_pickle(sys.argv[1])
