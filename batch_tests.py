from haardict import *
import numpy as np
import sys
import pandas as pd
import ipdb
import gc

np.random.seed(123)

def run_and_save(fpath = None):
    """
    Run batch tests. Returns list of Test instances and DatFrame with test results. If fpath is set pickles of these are saved. 
    """
    learnimgs = 'img/flowers_pool-rescale.npy'
    #learnimgs = 'img/boat512.png'
    #learnimgs = ['img/cameraman256.png','img/lena512.png','img/barbara512.png','img/peppers256.png']

    #codeimg = 'img/landscape2-rescaled.jpg'
    codeimg = 'img/flowers_pool-rescale.npy'
    #codeimg = 'img/cameraman256.png'

    #patch_sizes_npatches_dictsize = [((8,8),15000),((12,12),10000),((16,16),8000),((24,24),4000),((32,32),1500)]
    #patch_sizes_npatches = [((8,8),150),((12,12),100)]
    #patch_sizes_npatches = [((32,32),int(2e4))]
    #patch_sizes_npatches = [((8,8),int(2550))]
    npatches = np.arange(150,2e4,600).astype('int')
    patch_sizes_npatches = [((8,8),k) for k in npatches]
    patch_sizes_npatches += [((16,16),k) for k in npatches]
    #patch_sizes_npatches += [((32,32),k) for k in npatches[::3]]
    overlap = True
    df_saveable = Saveable()
    df_saveable.df = pd.DataFrame()
    #tests_saveable = Saveable()
    #tests_saveable.testlist = []
    if fpath is not None:
        dict_fpath = fpath+'dicts.pickle'
        df_fpath = fpath+'df.pickle'
    for i,psize_np in enumerate(patch_sizes_npatches):
        psize,npat = psize_np
        newtests = 0
        pdim = psize[0]*psize[1]
        dsize = max(1,min(int(pdim*1.5),int(npat/2))) #dictionary 50% bigger than dimension
        spars = max(1,min(int(pdim*0.01),int(npat/2))) #sparsity 1% of dimension
        cur_tests = []
        for meth in ['ksvd', 'haar-dict','centroids-dict']:
        #for meth in ['haar-dict','centroids-dict']:
            if meth == 'ksvd':
                ct = Test(learnimgs,npat,psize,noisevar=0,overlapped_patches=overlap)
                ct.debug = True
                ct.learn_dict(method=meth, dictsize=dsize, ksvdsparsity=3*spars)
                cur_tests.append(ct)
            else:
                cur_tests = []
                for clust in ['twomeans','twomaxoids']:
                #for clust in ['twomaxoids']:
                    ct = Test(learnimgs,npat,psize,noisevar=0,overlapped_patches=overlap)
                    ct.debug = True
                    ct.learn_dict(method=meth, dictsize=dsize, clustering=clust)
                    cur_tests.append(ct)
        for k in range(1,15):
            for t in cur_tests:
                t.overlapped_patches = False
                t.reconstruct(codeimg,k*spars)
                #tests_saveable.testlist.append(t)
                t._compute_test_results()
                t.print_results()
                df_saveable.df = df_saveable.df.append(t.test_results,ignore_index=True)
                if fpath is not None:
                    #tests_saveable.save_pickle(dict_fpath)
                    df_saveable.save_pickle(df_fpath)                    
                newtests += 1

            gc.collect()
    #return(tests_saveable.testlist,df_saveable.df)
    return(df_saveable.df)


def plot1(df, psize):
    #toplot = [v for k,v in df.fillna('NaN').groupby(['learning_method','clustering','similarity_measure'])]
    toplot = [v for k,v in df.fillna('NaN').groupby(['learning_method','clustering'])]
    for cur_df in toplot:
        cur_df = cur_df[cur_df['patch_size'] == psize]
        lab = cur_df.iloc[0]['learning_method']
        if lab in ['haar-dict','centroids-dict']:
            clust = cur_df.iloc[0]['clustering']
            lab += '-'+clust
            if clust == 'spectral':
                lab += '-'+cur_df.iloc[0]['similarity_measure']
        #series = cur_df.set_index('n.patches')['haarpsi']
        series = cur_df.set_index('reconstruction_sparsity')['haarpsi']
        #series = cur_df.set_index('n.patches')['learning_time']
        series.plot(label=lab,style='x--')
    plt.legend(loc='best')
    plt.show()
            
if __name__ == '__main__':
    run_and_save_pickle(sys.argv[1])
