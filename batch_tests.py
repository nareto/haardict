from haardict import *
import numpy as np
import sys
import pandas as pd
import ipdb

np.random.seed(123)

def run_and_save(fpath = None):
    """
    Run batch tests. Returns list of Test instances and DatFrame with test results. If fpath is set pickles of these are saved. 
    """
    #learnimgs = 'img/flowers_pool-rescale.npy'
    learnimgs = 'img/boat512.png'
    #learnimgs = ['img/cameraman256.png','img/lena512.png','img/barbara512.png','img/peppers256.png']

    #codeimg = 'img/landscape2-rescaled.jpg'
    #codeimg = 'img/flowers_pool-rescale.npy'
    codeimg = 'img/cameraman256.png'

    #patch_sizes_npatches_dictsize = [((8,8),15000),((12,12),10000),((16,16),8000),((24,24),4000),((32,32),1500)]
    #patch_sizes_npatches = [((8,8),150),((12,12),100)]
    npatches = np.arange(150,2e4,600).astype('int')
    patch_sizes_npatches = [((8,8),k) for k in npatches]
    overlap = False
    df_saveable = Saveable()
    df_saveable.df = pd.DataFrame()
    tests_saveable = Saveable()
    tests_saveable.testlist = []
    if fpath is not None:
        dict_fpath = fpath+'dicts.pickle'
        df_fpath = fpath+'df.pickle'
    for i,psize_np in enumerate(patch_sizes_npatches):
        psize,npat = psize_np
        newtests = 0
        cur_tests = []
        pdim = psize[0]*psize[1]
        dsize = int(pdim*1.5) #dictionary 50% bigger than dimension
        spars = int(pdim*0.05) #sparsity 5% of dimension
        #for meth in ['haar-dict','centroids-dict','ksvd']:
        for meth in ['haar-dict','centroids-dict']:
            if meth == 'ksvd':
                cur_test = Test(learnimgs,npat,psize,noisevar=0,overlapped_patches=overlap)
                cur_test.debug = True
                cur_test.learn_dict(method=meth, dictsize=dsize)
                cur_test.reconstruct(codeimg,spars)
                tests_saveable.testlist.append(cur_test)
                cur_test._compute_test_results()
                df_saveable.df = df_saveable.df.append(cur_test.test_results,ignore_index=True)
                if fpath is not None:
                    tests_saveable.save_pickle(dict_fpath)
                    df_saveable.save_pickle(df_fpath)                    
                newtests += 1
            else:
                for clust in ['twomeans','twomaxoids']:
                    cur_test = Test(learnimgs,npat,psize,noisevar=0,overlapped_patches=overlap)
                    cur_test.debug = True
                    cur_test.learn_dict(method=meth, dictsize=dsize, clustering=clust)
                    cur_test.reconstruct(codeimg,spars)
                    tests_saveable.testlist.append(cur_test)
                    cur_test._compute_test_results()
                    df_saveable.df = df_saveable.df.append(cur_test.test_results,ignore_index=True)
                    if fpath is not None:
                        tests_saveable.save_pickle(dict_fpath)
                        df_saveable.save_pickle(df_fpath)
                    newtests += 1
        #for k in range(newtests):
        #    ct = tests_saveable.testlist[-k-1]
        #    ct._compute_test_results()
        #    df_saveable.df = df_saveable.df.append(ct.test_results,ignore_index=True)
        #    if fpath is not None:
        #        df_saveable.save_pickle(df_fpath)
    return(tests_saveable.testlist,df_saveable.df)

def print_test_results(pickled_tests):
    tests = Saveable()
    tests.load_pickle(pickled_tests)
    for t in tests.testlist:
        t.print_results()


def plot1(df, psize):
    #toplot = [v for k,v in df.fillna('NaN').groupby(['learning_method','clustering','similarity_measure'])]
    toplot = [v for k,v in df.fillna('NaN').groupby(['learning_method','clustering'])]
    #return(toplot)
    for cur_df in toplot:
        lab = cur_df.iloc[0]['learning_method']
        if lab in ['haar-dict','centroids-dict']:
            clust = cur_df.iloc[0]['clustering']
            lab += '-'+clust
            if clust == 'spectral':
                lab += '-'+cur_df.iloc[0]['similarity_measure']
        cur_df.set_index('n.patches')['haarpsi'].plot(label=lab,style='x--')
        #cur_df.set_index('n.patches')['learning_time'].plot(label=lab,style='x--')
    plt.legend(loc='best')
    plt.show()
            
if __name__ == '__main__':
    run_and_save_pickle(sys.argv[1])
