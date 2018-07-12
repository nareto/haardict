import ipdb
import itertools
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as sslinalg
import skimage.io
import skimage.color
from skimage.filters import threshold_otsu
import scipy.sparse
from scipy.spatial.distance import pdist
import pyemd
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import ward_tree
from sklearn.neighbors import kneighbors_graph
import oct2py
import gc
import queue
import pywt
import datetime as dt
#from oct2py import octave
#octave.eval('pkg load image')
#oc = oct2py.Oct2Py()
##octave.addpath('ksvd')
#octave.addpath('ksvdbox/')
#octave.addpath('ompbox/')
from haarpsi import haar_psi

_METHODS_ = ['2ddict','ksvd']
_CLUSTERINGS_ = ['2means','spectral']
_TRANSFORMS_ = ['2dpca','wavelet','wavelet_packet','shearlets']
WAVPACK_CHARS = 'adhv'
_MIN_SIGNIFICATIVE_MACHINE_NUMBER = 1e-3
_MAX_SIGNIFICATIVE_MACHINE_NUMBER = 1e3

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

def sumsupto(k):
    return(k*(k+1)/2)

def matrix2patches(matrix,shape=None):
    """Returns list of arrays obtained from columns of matrix"""
    m,n = matrix.shape
    if shape is None:
        size = int(np.sqrt(m))
        shape = (size,size)
    patches = []
    for col in matrix.transpose():
        patches.append(col.reshape(shape))
    return(patches)

def patches2matrix(patches):
    """Returns matrix with columns given by flattened arrays in input"""
    m = patches[0].size
    matrix = np.hstack([p.reshape(m,1) for p in patches])
    return(matrix)

def read_raw_img(img,as_grey=True):
    import rawpy
    raw = rawpy.imread(img)
    rgb = raw.postprocess()
    if not as_grey:
        return(rgb)
    else:
        #gray = (rgb[:,:,0] + rgb[:,:,1] + rgb[:,:,2])/3
        gray = skimage.color.rgb2gray(rgb)
        return(gray)

def rescale(img,useimgextremes=False,oldmin=0,oldmax=1,newmin=0,newmax=255):
    """Linearly rescales values in img from newmin to newmax"""

    if useimgextremes:
        curmin,curmax = img.min(),img.max()
    else:
        curmin,curmax = oldmin,oldmax
    angcoeff = (newmax-newmin)/(curmax-curmin)
    f = lambda x: newmax+angcoeff*(x-curmax)
    out = np.zeros_like(img)
    for idx,val in np.ndenumerate(img):
        out[idx] = f(val)
    return(out)

def clip(img):
    """Clips values in img to be between 0 and 255"""
    
    out = img.copy()
    out[out < 0] = 0
    out[out > 255] = 255
    return(out)

def stack(mat):
    out = np.hstack(np.vsplit(mat,mat.shape[0])).transpose()
    return(out)

def psnr(img1,img2):
    """Computes PSNR of img1 vs. img2"""
    
    mse = np.sum((img1 - img2)**2)
    if mse == 0:
        return(-1)
    mse /= img1.size
    mse = np.sqrt(mse)
    maxval = img1.max()
    return(20*np.log10(maxval/mse))

def HaarPSI_octave(img1,img2):
    """Computes HaarPSI of img1 vs. img2. Requires file HaarPSI.m to be present in working directory"""
    #if 'octpy2' not in dir():
    from oct2py import octave
    octave.eval('pkg load image')
    img1 = img1.astype('float64')
    img2 = img2.astype('float64')
    haarpsi = octave.HaarPSI(img1,img2,0)
    return(haarpsi)

#dissimilarity measures 
def simmeasure_haarpsi(reshape=False):
    sim_meas = lambda patch1,patch2: haar_psi(255*patch1,255*patch2)[0]
    def dh(p1,p2):
        if reshape is not False:
            p1 = p1.reshape(reshape)
            p2 = p2.reshape(reshape)
        return(sim_meas(p1,p2))
    return(dh)

#beta=0.06
def simmeasure_frobenius(beta=0.06,samples=None):
    if samples is None:
        datavar = 1
    else:
        avg = sum(samples)/len(samples)
        datavar = sum([np.linalg.norm(p-avg,ord='fro')**2 for p in samples])/len(samples)
    ret = lambda patch1,patch2: np.exp(-beta*(np.linalg.norm(patch1 - patch2,ord='fro')**2)/datavar)
    return(ret)

def simmeasure_emd(patch_size,beta=0.06,samples=None):
    prows,pcols = patch_size
    histlength = prows*pcols
    metric_matrix = np.zeros((histlength,histlength))
    for i,j in itertools.combinations(range(histlength),2):
        row1,col1 = int(i/pcols),i%pcols
        row2,col2 = int(j/pcols),j%pcols
        p1 = np.array([row1,col1])
        p2 = np.array([row2,col2])
        metric_matrix[i,j] = np.linalg.norm(p1-p2)
    metric_matrix += metric_matrix.transpose()
    if samples is None:
        datavar = 1
    else:
        avg = sum(samples)/len(samples)
        datavar = sum([pyemd.emd(p.astype('float64').flatten(),avg.astype('float64').flatten(),metric_matrix)**2 for p in samples])/len(samples)
    ret = lambda patch1,patch2: np.exp(-beta*(pyemd.emd(patch1.astype('float64').flatten(),patch2.astype('float64').flatten(),metric_matrix)**2)/datavar)
    return(ret)

def centroid(values):
    """Computes mean of 'values'"""

    if len(values) == 0:
        raise Exception("Can't compute centroid of void set")
    centroid = 0
    for val in values:
        centroid += val
    centroid /= len(values)
    return(centroid)

def pywt2array(coeffs):
    levels = len(coeffs) - 1
    baseexp = np.log2(coeffs[0].shape[0])
    tot_length = int(np.round(4**baseexp + sum([3*4**(baseexp+i) for i in range(levels)])))
    out = np.zeros(tot_length)
    out[:int(np.round(4**baseexp))] = coeffs[0].flatten()
    offset = int(np.round(4**baseexp))
    for lev,details in enumerate(coeffs[1:]):
        dh,dv,dd = details
        length = int(np.round(4**(baseexp+lev)))
        for d in details:
            out[offset:offset+length] = d.flatten()
            offset += length
    return(out)

def array2pywt(array,levels):
    baseexp = int(np.log2(array.size)/2)
    l = baseexp-levels
    out = [array[:4**l].reshape(2**l,2**l)]
    offset = 4**(baseexp-levels)
    for l in range(baseexp-levels,baseexp):
        shape = (2**l,2**l)
        size = shape[0]**2
        out.append((array[offset:offset+size].reshape(shape),\
                    array[offset+size:offset+2*size].reshape(shape),\
                    array[offset+2*size:offset+3*size].reshape(shape)))
        offset += 3*size
    return(out)

def wavpack2array(wp):
    values = []
    for string in itertools.product(WAVPACK_CHARS,repeat=wp.maxlevel):
        key = "".join(string)
        val = wp[key].data
        values.append(val)
    return(np.array(values).flatten())

def array2wavpack(array,wavelet,levels,mode='periodic'):
    wp = pywt.WaveletPacket2D(None,wavelet,mode,maxlevel=levels)
    i = 0
    for string in itertools.product(WAVPACK_CHARS,repeat=wp.maxlevel):
        key = "".join(string)
        val = array[i].reshape(1,1)
        wp[key] = val
        i += 1
    return(wp)

def normalize_matrix(matrix, norm_order=2):
    """==  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================

    The nuclear norm is the sum of the singular values."""
    cent_matrix = matrix - matrix.mean(axis=0)
    ncols = matrix.shape[1]
    normalization_coefficients = np.ones(shape=(matrix.shape[1],))
    for j in range(ncols):
        col = cent_matrix[:,j]
        norm = np.linalg.norm(col,ord=norm_order)
        if norm != 0:
            col /= norm
            normalization_coefficients[j] = norm
    return(cent_matrix,normalization_coefficients)

def twomeansval(values,k):
    """Computes value of 1D 2-means minimizer function for clusters values[:k] and values[k:]"""
    
    C1 = values[:k]
    C2 = values[k:]
    c1bar = centroid(C1)
    c2bar = centroid(C2)
    dist = np.abs(c1bar - c2bar)
    val = 0
    for s in C1:
        val += (s-c1bar)**2
    for s in C2:
        val += (s-c2bar)**2
    return(val,dist)

def is_sorted(values):
    """Returns True if values is sorted decreasingly, False otherwise"""
    
    prev_v = values[0]
    for v in values[1:]:
        if v < prev_v:
            return(False)
    return(True)

def oneDtwomeans(values):
    """Computes 2-means value of 1D data in values, which must be ordered increasingly"""
    
    best_kval = None
    best_idx = None
    centroid_dist = None
    prev_val = values[0]
    #for separating_idx in range(1,len(values)):
    for separating_idx in range(1,len(values)):
        val = values[separating_idx]
        if val < prev_val:
            raise Exception("Input list must be sorted")
        prev_val = val
        kval,dist = twomeansval(values,separating_idx)
        if best_kval is None or kval < best_kval:
            best_kval = kval
            best_idx = separating_idx
            centroid_dist = dist
    
    return(best_idx,best_kval,centroid_dist)

def atoms_prob(coef_matrix):
    """Sums row of matrix and normalizes"""
    ret = np.array([np.abs(row).sum() for row in coef_matrix])
    ret /= ret.sum()
    return(ret)

def entropy(array):
    """Returns entropy of normalized array"""
    ent = 0
    for a in array:
        if a != 0:
            ent += a*np.log2(a)
    return(-ent)

def extract_patches(array,size=(8,8)):
    """Returns list of small arrays partitioning the input array. See also assemble_patches"""
    
    ret = []
    height,width = array.shape
    vstep,hstep = size
    for j in range(0,width-hstep+1,hstep):
        for i in range(0,height-vstep+1,vstep):
            subimg = array[i:i+vstep,j:j+hstep]
            ret.append(subimg)
    return(ret)

def assemble_patches(patches,out_size):
    """Returns an array given by row-stacking the arrays in patches. See also extract_patches"""
    
    height,width = out_size
    out = np.zeros(shape=out_size)
    idx = 0
    vstep,hstep = patches[0].shape
    for j in range(0,width-hstep+1,hstep):
        for i in range(0,height-vstep+1,vstep):
            #j = np.random.randint(0,width-hstep+1)
            #i = np.random.randint(0,height-vstep+1)
            out[i:i+vstep,j:j+hstep] = patches[idx]
            idx += 1
    return(out)
    
def low_rank_approx(svdtuple=None, A=None, r=1):
    """
    Returns r-rank approximation of matrix A given by its SVD
    """
    if svdtuple is None:
        svdtuple = np.linalg.svd(A, full_matrices=False)
    u, s, v = svdtuple
    ret = np.zeros((u.shape[0],v.shape[1]))
    for i in range(r):
        ret += s[i] * np.outer(u.T[i], v[i])
    return(ret)

def affinity_matrix2(samples,similarity_measure,threshold):
    X = patches2matrix(samples).T
    return(pdist(X,similarity_measure))

def affinity_matrix(samples,similarity_measure,threshold,symmetric=True):
    """Returns sparse matrix representation of matrix of pairwise similarities, keeping only the pairwise similarities that are below the given threshold"""

    nsamples = len(samples)
    print('Computing affinity matrix...')

    data = []
    rows = []
    cols = []
    counter = 0
    for i,j in itertools.combinations(range(nsamples),2):
        print('\r%d/%d' % (counter + 1,sumsupto(nsamples-1)), sep=' ',end='',flush=True)
        d = similarity_measure(samples[i], samples[j])
        if d > threshold:
            data.append(d)
            rows.append(i)
            cols.append(j)
        counter += 1
    print('\n %.2f percentage of the similarities is above the threshold' % (100*len(data)/sumsupto(len(samples))))
    data = np.array(data)
    #data,rows,cols = _compute_diss_rows_cols(diss)
    #thresholded_data = data.copy()
    #data.sort()
    #thresh = data[int(affinity_matrix_threshold_perc*len(data))]
    #removed = 0
    #for i,d in enumerate(data):
    #    if d > thresh:
    #        idx = i-removed
    #        thresholded_data = thresholded_data[:idx] + thresholded_data[idx+1:]
    #        rows = rows[:idx] + rows[idx+1:]
    #        cols = cols[:idx] + cols[idx+1:]
    #        removed += 1
    #data = np.array(thresholded_data)
    #diss_data = data
    #avgd = data.mean()
    #vard = np.sqrt(data.var())
    #beta = -(np.log(_MIN_SIGNIFICATIVE_MACHINE_NUMBER*_MAX_SIGNIFICATIVE_MACHINE_NUMBER))/(2*(avgd - 5*vard))
    #beta = -(np.log(_MIN_SIGNIFICATIVE_MACHINE_NUMBER))/(avgd - 3*vard)
    #print('\navgd = %f\n vard = %f\n beta = %f' % (avgd,vard,beta))
    #data /= data.max()
    #beta = 4
    #exp_data = np.exp(-beta*data)
    #expaff_mat = csr_matrix((exp_data,(rows,cols)),shape=(nsamples,nsamples))
    #print('...done')
    #expaff_mat = expaff_mat + expaff_mat.transpose()
    #affinity_matrix = expaff_mat
    #print(len(affinity_matrix.data)/(affinity_matrix.shap
    #return(expaff_mat)

    aff_mat = csr_matrix((data,(rows,cols)),shape=(nsamples,nsamples))
    if symmetric:
        aff_mat = aff_mat + aff_mat.transpose() #make it symmetrical
    #print(len(affinity_matrix.data)/(affinity_matrix.shap
    return(aff_mat,data,rows,cols)
    
    

def covariance_matrix(patches):
    """Computes generalized covariance matrix of arrays in patches"""

    centroid = sum(patches)/len(patches)
    p = patches[0]
    ret = np.zeros_like(np.dot(p.transpose(),p))
    for patch in patches:
        addend = np.dot((patch - centroid).transpose(),(patch-centroid))
        ret += addend
    ret /= len(patches)
    return(ret)

def orgmode_table_line(strings_or_n):
    out ='| ' + ' | '.join([str(s) for s in strings_or_n]) + ' |'
    return(out)

def np_or_img_to_array(path,crop_to_patchsize=None):
    if path[-3:].upper() in  ['JPG','GIF','PNG','EPS']:
        ret = skimage.io.imread(path,as_grey=True).astype('float64')/255
    elif path[-3:].upper() in  ['NPY']:
        ret = np.load(path)
    elif path[-4:].upper() == 'TIFF' or path[-3:].upper() == 'CR2':
        ret = read_raw_img(path)
    if crop_to_patchsize is not None:
        m,n = crop_to_patchsize
        M,N = ret.shape
        ret = ret[:M-(M%m),:N-(N%n)]
    return(ret)

def show_or_save_patches(patchlist,rows,cols,savefile=None):
    #fig, axis = plt.subplots(rows,cols,sharex=True,sharey=True,squeeze=True)
    #fig, axis = plt.subplots(rows,cols,squeeze=True)
    fig, axis = plt.subplots(rows,cols)
    for idx,ax in np.ndenumerate(axis):
        try:
            i = idx[0]
        except IndexError:
            i = 0
        try:
            j = idx[1]
        except IndexError:
            j = 0
        if len(idx) == 1:
            i = 0
            j = idx[0]
        ax.set_axis_off()
        ax.imshow(patchlist[cols*i + j],interpolation='nearest')
    #hacks to try and get nice subplots
    #plt.tight_layout(pad=0.1,h_pad=0.1,w_pad=0.1)
    #plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.1,hspace=0.1)
    plt.subplots_adjust(left=0.1,right=0.45,bottom=0.1,top=0.9,wspace=0.1,hspace=0.1)
    #plt.subplot_tool()
    
    if savefile is None:
        plt.show()
    else:
        plt.savefig(savefile)

def learn_dict(paths,npatches=None,patch_size=(8,8),method='2ddict',dictsize=None,clustering='2means',cluster_epsilon=2,spectral_similarity='haarpsi',simmeasure_beta=0.06,affinity_matrix_threshold=1,ksvdsparsity=2,transform=None,twodpca_l=3,twodpca_r=3,wav_lev=3,dict_with_transformed_data=False,wavelet='haar',dicttype='haar'):
    """Learns dictionary based on the selected method. 

    paths: list of paths of images to learn the dictionary from
    npatches: if an int, only this number of patches (out of the complete set extracted from the learning images) will be used for learning
    patch_size: size of the patches to be extracted
    method: the chosen method. The possible choices are:
    	- 2ddict: 2ddict procedure 
    	- ksvd: uses the KSVD method
    transform: whether to transform the data before applying the method:
    	- 2dpca: applies 2DPCA transform (see options: twodpca_l,twodpca_r)
    	- wavelet: applies wavelet transform to patches - see also wav_lev, wavelet
    	- wavelet_packet: appliest wavelet_packet transform to patches - see also wavelet
    clustering: the clustering used (only for 2ddict method):
    	- 2means: 2-means on the vectorized samples
    	- spectral: spectral clustering (slow)
    cluster_epsilon: threshold for clustering (lower = finer clustering)
    spectral_similarity: similarity measure used for spectral clustering. Can be 'frobenius','haarpsi' or 'emd' (earth's mover distance)
    simmeasure_beta: beta parameter for Frobenius and Earth Mover's distance similarity measures
    affinity_matrix_threshold: threshold for pairwise similarities in affinity matrix
    twodpca_l,twodpca_r: number of left and right feature vectors used in 2DPCA; the feature matrices will be twodpca_l x twodpca_r
    wavelet: type of wavelet for wavelet or wavelet_packet transformations. Default: haar
    wav_lev: number of levels for the wavelet transform
    dict_with_transformed_data: if True, the dictionary will be computed using the transformed data instead of the original patches
    """

    if method not in _METHODS_:
        raise Exception("'method' has to be on of %s" % _METHODS_)
    if clustering not in _CLUSTERINGS_:
        raise Exception("'clustering' has to be on of %s" % _CLUSTERINGS_)
    if transform is not None and transform not in _TRANSFORMS_:
        raise Exception("'transform' has to be on of %s" % _TRANSFORMS_)
    images = []
    for f in paths:
        images.append(np_or_img_to_array(f,patch_size))

    patches = []
    for i in images:
        patches += [p for p in extract_patches(i,patch_size)]
    #patches = np.array(patches)
    if npatches is not None:
        patches = [patches[i] for i in np.random.permutation(range(len(patches)))][:npatches]
    print('Working with %d patches' % len(patches))

    tic()
    #TRANSFORM
    if transform == '2dpca':
        transform_instance = twodpca_transform(patches,twodpca_l,twodpca_r)
    elif transform == 'wavelet':
        transform_instance = wavelet_transform(patches,wav_lev,wavelet)
    elif transform == 'wavelet_packet':
        transform_instance = wavelet_packet(patches,wavelet)
    elif transform is None:
        transform_instance = dummy_transform(patches)
    data_to_cluster = transform_instance.transform() #tuple of transformed patches
    
    #CLUSTER
    if clustering is None or method != '2ddict':
        cluster_instance = dummy_clustering(data_to_cluster)
    elif clustering == '2means':
        cluster_instance = twomeans_clustering(data_to_cluster,nbranchings=dictsize,epsilon=cluster_epsilon)
    elif clustering == 'spectral':
        cluster_instance = spectral_clustering(data_to_cluster,epsilon=cluster_epsilon,similarity_measure=spectral_similarity,simmeasure_beta=simmeasure_beta,affinity_matrix_threshold=affinity_matrix_threshold)    
    cluster_instance.compute()

    #BUILD DICT
    #patches4dict = patches
    patches4dict = data_to_cluster
    if dict_with_transformed_data:
        patches4dict = data_to_cluster
    if method == '2ddict':
        dictionary = hierarchical_dict(cluster_instance,patches4dict,dicttype)
    elif method == 'ksvd':
        dictionary = ksvd_dict(patches4dict,dictsize=dictsize,sparsity=ksvdsparsity)
    if method == '2ddict':
        dictionary.haar_dictelements = transform_instance.reverse(dictionary.haar_dictelements)
        dictionary.centroid_dictelements = transform_instance.reverse(dictionary.centroid_dictelements)
        dictionary.set_dicttype(dicttype)
    else:
        dictionary.dictelements = transform_instance.reverse(dictionary.dictelements)
        dictionary._dictelements_to_matrix()
    time = toc(False)
    return(dictionary,time)

def reconstruct(oc_dict,imgpath,psize,sparsity=5):
    clip = False
    spars= sparsity
    img = np_or_img_to_array(imgpath,psize)
    patches = extract_patches(img,psize)

    tic()
    outpatches = []
    means,coefs = oc_dict.encode_patches(patches,spars)
    rec_patches = oc_dict.reconstruct_patches(coefs,means)
    
    #reconstructed_patches = [p.reshape(psize) for p in rec_matrix.transpose()]
    reconstructed = assemble_patches(rec_patches,img.shape)
    time = toc(False)
    if clip:
        reconstructed = clip(reconstructed)
    return(reconstructed,coefs,time)        

class Saveable():
    def save_pickle(self,filepath):
        f = open(filepath,'wb')
        pickle.dump(self.__dict__,f,3)
        f.close()

    def load_pickle(self,filepath):
        f = open(filepath,'rb')
        tmpdict = pickle.load(f)
        f.close()
        self.__dict__.update(tmpdict)
    
class Node():
    def __init__(self,samples_idx,parent,isleftchild=True):
        self.parent = parent
        if parent is None:
            self.depth = 0
            self.idstr = ''
        else:
            self.depth = self.parent.depth + 1
            if isleftchild:
                self.idstr = parent.idstr + '0'
            else:
                self.idstr = parent.idstr + '1'
        self.children = None
        #self.samples_idx = None
        self.samples_idx = tuple(samples_idx) #indexes of d.samples_idx where d is an ocdict
        self.nsamples= len(self.samples_idx)
        #if samples_idx is not None:
        #    self.samples_idx = tuple(samples_idx) #indexes of d.samples_idx where d is an ocdict
        #    self.nsamples= len(self.samples_idx)

    #def samples_list(self,ocdict):
    #    return([ocdict.samples[i] for i in self.samples_idx])

class Transform(Saveable):
    """Transform for 2D data. Input: list of patches"""
    def __init__(self,patch_list):
        self.patch_list = patch_list
        
    def transform(self):
        """Computes the transform and returns a list of patches of the transformed data"""
        return(self._transform())

    def reverse(self,patches):
        """Computes the transform and returns a list of patches of the transformed data"""
        return(self._reverse_transform(patches))

class Cluster(Saveable):
    """Hierarchical cluster of data"""

    def __init__(self,samples):
        self.samples = samples
        self.dim = self.samples[0].size
        self.nsamples = len(samples)
        self.root_node = None

    def _compute_affinity_matrix(self,**args):
        if self.similarity_measure == 'haarpsi':
            sim = simmeasure_haarpsi()
        elif self.similarity_measure == 'frobenius':
            sim = simmeasure_frobenius(beta=self.simmeasure_beta,samples=self.samples)
        elif self.similarity_measure == 'emd':
            sim  = simmeasure_emd(self.patch_size,beta=self.simmeasure_beta,samples=self.samples)

        self.affinity_matrix,self.affmat_data,self.affmat_rows,self.affmat_cols = affinity_matrix(self.samples,sim,self.affinity_matrix_threshold,args)
        #print(len(self.affinity_matrix.data)/(self.affinity_matrix.shape[0]*self.affinity_matrix.shape[1]))

    def compute(self):
        self._cluster()

    def get_node(self,idstr):
        revid = list(idstr[::-1])
        cur = self.root_node
        nextn = cur.children[int(revid.pop())]
        while len(revid) >= 1:
            cur = nextn
            try:
                nextn = cur.children[int(revid.pop())]
            except TypeError:
                raise Exception('revid string is too long')
        return(nextn)
    
    def subtree(self,startingnode): #TODO: test method
        patches = []
        curn = self.root_node
        for c in startingnode:
            nextn = curn.children[int(c)]
            curn = nextn
        tovisit = [nextn]
        while len(tovisit) > 0:
            cur = tovisit.pop()
            patches.append(tuple([self.patches[i] for i in cur.patches_idx]))
            if cur.children is not None:
                tovisit = [cur.children[0]] + tovisit
                tovisit = [cur.children[1]] + tovisit
        return(patches)
        
    def show_clusters(self,shape=None,startingnode=None,savefile=None):
        clusters = []
        if startingnode is None:
            tovisit = [self.root_node]
        else:
            curn = self.root_node
            for c in startingnode:
                nextn = curn.children[int(c)]
                curn = nextn
            tovisit = [nextn]
        while len(tovisit) > 0:
            cur = tovisit.pop()
            centroid = sum([self.samples[i] for i in cur.samples_idx[1:]],self.samples[cur.samples_idx[0]])
            #centroid = sum([self.patches[i] for i in cur.patches_idx])
            centroid /= len(cur.samples_idx)
            if cur.idstr == '':
                string = 'root'
            else:
                string = cur.idstr
            clusters.append((centroid,string))
            if cur.children is not None:
                tovisit = [cur.children[0]] + tovisit
                tovisit = [cur.children[1]] + tovisit
        if shape is None:
            l = int(np.sqrt(len(clusters)))
            shape = (min(10,l),min(10,l))
        rows,cols = shape
        fig, axis = plt.subplots(rows,cols,sharex=True,sharey=True,squeeze=True)
        if len(axis.shape) == 2:
            for idx, a in np.ndenumerate(axis):
                a.set_axis_off()
                clust,string = clusters[rows*idx[0] + idx[1]]
                a.imshow(clust,interpolation='nearest')
                a.text(0,0,string,color='red')
        else:
            for idx, a in np.ndenumerate(axis):
                a.set_axis_off()
                clust,string = clusters[idx[0]]
                a.imshow(clust,interpolation='nearest')
                a.text(0,0,string,color='red')
        if savefile is None:
            plt.show()
        else:
            plt.savefig(savefile)
        
class Oc_Dict(Saveable):
    """Dictionary"""
    def __init__(self,matrix,patches_dim=None):
        self.matrix = matrix
        self.shape = matrix.shape
        self.atom_dim,self.cardinality = matrix.shape
        self.patches_dim = patches_dim
        if patches_dim is not None:
            self.twod_dict = True

    def compute(self):
        #self.compute_dictelements()
        self._compute()
        self._dictelements_to_matrix()
        
    def _dictelements_to_matrix(self):
        self.matrix = np.vstack([x.flatten() for x in self.dictelements]).transpose()
        self.matrix,self.normalization_coefficients = normalize_matrix(self.matrix)

    def max_similarities(self,similarity_measure='frobenius',progress=True):
        """Computes self.max_sim, array of maximum similarities between dictionary atoms and input patches"""
        
        if similarity_measure == 'haarpsi':
            sim = simmeasure_haarpsi()
        elif similarity_measure == 'frobenius':
            sim = simmeasure_frobenius()
        elif similarity_measure == 'emd':
            sim = simmeasure_emd(self.patch_size)
        max_sim = []
        counter = 0
        if progress:
            print('Computing minimum similarities with similarity_measure measure: %s' % (similarity_measure))
        for d in self.dictelements:
            md = sim(d,self.patches[0])
            for p in self.patches:
                if progress:
                    print('\r%d/%d' % (counter + 1,len(self.patches)*self.cardinality), sep=' ',end='',flush=True)
                md = min(md,sim(d,p))
                counter += 1
            max_sim.append(md)
        self.max_sim = max_sim
    
    def mutual_coherence(self,progress=True):
        self.max_cor = 0
        self.argmax_cor = None
        ncols = self.matrix.shape[1]
        if progress:
            print('Computing mutual coherence of dictionary')
        counter = 0
        for j1,j2 in itertools.combinations(range(ncols),2):
            if progress:
                print('\r%d/%d' % (counter + 1,sumsupto(ncols-1)), sep=' ',end='',flush=True)
            col1 = self.matrix[:,j1]
            col2 = self.matrix[:,j2]
            sp = np.dot(col1,col2)
            sp /= np.linalg.norm(col1)*np.linalg.norm(col2)
            if sp > self.max_cor:
                self.argmax_cor = (j1,j2)
                self.max_cor = sp
            counter += 1
        #print('\nMaximum correlation: %f ---  columns %d and %d' % (self.max_cor,self.argmax_cor[0],self.argmax_cor[1]))

    #def _encode_sample(self, sample):
    #    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
    #    #outnorm = np.linalg.norm(sample)
    #        #matrix = self.matrix
    #    #normalize = self.matrix_is_normalized
    #    #if normalize:
    #    #    #sample /= outnorm
    #    #    matrix = self.normalized_matrix
    #    #    norm_coefs = self.normalization_coefficients
    #    omp.fit(self.matrix,sample)
    #    return(omp._coef)        

    def encode_samples(self,samples,sparsity,center_samples=True):
        #if hasattr(self,'_encode'):
        #    self._encode(samples)
        means = []
        for s in samples.transpose():
            mean = s.mean()
            s -= mean
            means.append(mean)
        if isinstance(self,ksvd_dict) and self.useksvdencoding:
            print('\n\nUSING KSVD ENCODING\n\n\n')
            #ipdb.set_trace()
            coefs = self.encoding
        else:
            #omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity,fit_intercept=True)
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
            #if normalize:
            #    #sample /= outnorm
            #    matrix = self.normalized_matrix
            #    norm_coefs = self.normalization_coefficients
            omp.fit(self.matrix,samples)
            coefs = omp.coef_.transpose()
        return(means,coefs)

    def encode_patches(self,patches,sparsity,center_samples=True):
        return(self.encode_samples(patches2matrix(patches),sparsity,center_samples))
        
    def reconstruct_samples(self,coefficients,means=None):
        if hasattr(self,'_reconstruct'):
            self._reconstruct(coefficients,means)
        else:
            reconstructed = np.dot(self.matrix,coefficients)#.transpose())
            if means is not None:
                for i,m in enumerate(means):
                    reconstructed[:,i] += m
        return(reconstructed)

    def reconstruct_patches(self,coefficients,means=None):
        return(matrix2patches(self.reconstruct_samples(coefficients,means)))
    
    def encode_ompext(self,input_patch,sparsity,ompbox=True): #TODO: test method
        from oct2py import octave
        if ompbox:
            octave.addpath('ompbox')

        if input_patch.shape != self.patches[0].shape:
            raise Exception("Input patch is not of the correct size")
        if not self.matrix_computed:
            self._compute_matrix()
            
        y = input_patch.flatten()
        mean = np.mean(y)
        y -= mean
        matrix = self.normalized_matrix
        y = y.reshape(len(y),1)
        y  = y.astype('float64')
        matrix = matrix.astype('float64')
        #coef = octave.omp(matrix,y,matrix.transpose().dot(matrix),sparsity).todense()
        if ompbox:
            coef = octave.omp(matrix,y,np.array([]),sparsity).todense()
        else:
            coef = octave.OMP(sparsity,y.transpose(),matrix).transpose()
        return(coef,mean)

    def show_most_used_atoms(self,coefs,natoms = 100,savefile=None):
        if natoms < 15:
            rows,cols = (1,natoms)
        else:
            l = int(np.sqrt(natoms))
            rows,cols = (min(10,l),min(10,l))
        patches = []
        probs = atoms_prob(coefs)
        maxidx = probs.argsort()[::-1]
        for i in range(natoms):
            patches += [self.dictelements[maxidx[i]]]
        show_or_save_patches(patches,rows,cols,savefile=savefile)
            
    def show_dict_patches(self,shape=None,patch_shape=None,savefile=None):
        if patch_shape is None:
            s = int(np.sqrt(self.atom_dim))
            patch_shape = (s,s)
        if shape is None:
            l = int(np.sqrt(self.cardinality))
            shape = (min(10,l),min(10,l))
        rows,cols = shape
        if self.cardinality < 15:
            rows,cols = (1,self.cardinality)
        #if not hasattr(self,'atom_patches'):
        self.atom_patches = [col.reshape(patch_shape) for col in self.matrix.transpose()]
        #self.atoms_by_var = sorted(self.atom_patches,key=lambda x: x.var(),reverse=False)[:rows*cols]
        show_or_save_patches(self.atom_patches,rows,cols,savefile=savefile)

class dummy_transform(Transform):
    """Place holder transform which doesn't do anything"""
    
    def __init__(self,patch_list):
        Transform.__init__(self,patch_list)

    def _transform(self):
        return(self.patch_list)

    def _reverse_transform(self,patches):
        return(patches)

class twodpca_transform(Transform):
    """Transform that computes 2DPCA and feature matrices """
    
    def __init__(self,patch_list,l,r):
        self.l = l
        self.r = r
        Transform.__init__(self,patch_list)
        self._compute_simple_bilateral_2dpca()

    def _compute_horizzontal_2dpca(self):
        cov = covariance_matrix([p for p in self.patch_list])
        eigenvalues,U = sslinalg.eigs(cov,self.l)
        self.horizzontal_eigenvalues = eigenvalues
        self.U = U
        
    def _compute_vertical_2dpca(self):
        cov = covariance_matrix([p.transpose() for p in self.patch_list])
        eigenvalues,V = sslinalg.eigs(cov,self.r)
        self.vertical_eigenvalues = eigenvalues
        self.V = V
        
    def _compute_simple_bilateral_2dpca(self):
        self._compute_horizzontal_2dpca()
        self._compute_vertical_2dpca()
    
    def _compute_feature_matrix(self,patch):
        return(np.dot(np.dot(self.V.transpose(),patch),self.U))

    def _invert_feature_matrix(self,fmat):
        return(np.dot(np.dot(self.V,fmat),self.U.transpose()))

    def _transform(self):
        self.transformed_patches = tuple([self._compute_feature_matrix(p) \
                                          for p in self.patch_list])
        return(self.transformed_patches)

    def _reverse_transform(self,transf_patches=None):
        if transf_patches is None:
            transf_patches = self.transformed_patches
        outp = tuple([self._invert_feature_matrix(fmat) for fmat in transf_patches])
        return(outp)

        
class wavelet_transform(Transform):
    def __init__(self,patch_list,levels,wavelet='haar'):
        self.levels = levels
        self.wavelet = wavelet
        Transform.__init__(self,patch_list)
        
    def _transform(self):
        outshape = self.patch_list[0].shape
        self.transformed_patches = tuple([pywt2array(pywt.wavedec2(p,self.wavelet,'periodic',self.levels)).reshape(outshape) \
                                          for p in self.patch_list])
        return(self.transformed_patches)

    def _reverse_transform(self,transf_patches=None):
        if transf_patches is None:
            transf_patches = self.transformed_patches
        outp = [pywt.waverec2(array2pywt(p.flatten(),self.levels),self.wavelet,'periodic') for p in transf_patches]
        return(tuple(outp))

class wavelet_packet(Transform):
    def __init__(self,patch_list,wavelet='haar'):
        self.wavelet = wavelet
        Transform.__init__(self,patch_list)
        
    def _transform(self):
        outshape = self.patch_list[0].shape
        outp = []
        for p in self.patch_list:
            wp = pywt.WaveletPacket2D(p,self.wavelet,'periodic')
            outp.append(wavpack2array(wp).reshape(outshape))
        self.levels = wp.maxlevel
        self.transformed_patches = tuple(outp)
        return(self.transformed_patches)

    def _reverse_transform(self,transf_patches=None):
        if transf_patches is None:
            transf_patches = self.transformed_patches
        outp = []
        for p in transf_patches:
            outp.append(array2wavpack(p.flatten(),self.wavelet,self.levels).reconstruct())
        return(tuple(outp))
    
class dummy_clustering(Cluster):
    def _cluster(self):
        pass
    
class monkey_clustering(Cluster):
    """Randomly clusters the data. For testing purposes"""

    def __init__(self,samples):
        Cluster.__init__(self,samples)
    
    def _cluster(self,levels=3):
        self.tree_depth = levels
        self.root_node = Node(tuple(np.arange(self.nsamples)),None)
        tovisit = [self.root_node]
        while len(tovisit) > 0 and depth <= levels:
            cur = tovisit.pop()
            indexes = list(cur.samples_idx)
            np.random.shuffle(indexes)
            k = int(len(indexes)/2)
            lindexes = indexes[:k]
            rindexes = indexes[k:]
            lnode = Node(lindexes,cur,True)
            rnode = Node(rindexes,cur,False)
            cur.children = (lnode,rnode)
            tovisit = [lnode] + tovisit
            tovisit = [rnode] + tovisit
            depth = max((depth,lnode.depth,rnode.depth))
        self.leafs = tovisit
        self.tree_depth = depth
        self.tree_sparsity = len(self.leafs)/2**self.tree_depth
    
class twomeans_clustering(Cluster):
    """Clusters data using recursive 2-means"""

    def __init__(self,samples,nbranchings=None,epsilon=None,minsamples=5):
        Cluster.__init__(self,samples)
        if (nbranchings is None and epsilon is None) or (nbranchings is not None and epsilon is not None):
            raise Exception('Exactly one of nbranchings or epsilon has to be set')
        self.epsilon = epsilon
        self.nbranchings = nbranchings
        self.minsamples = minsamples
        self.cluster_matrix = patches2matrix(self.samples).transpose()

    def _cluster(self):
        self._cluster_wcss()
        
    def _cluster_wcss(self):
        self.root_node = Node(tuple(np.arange(self.nsamples)),None)
        if self.nbranchings is not None:
            self.visit = 'priority'
            tovisit = queue.PriorityQueue()
        else:
            self.visit = 'fifo'
            tovisit = queue.Queue()
            #tovisit = queue.LifoQueue()
        if self.visit == 'priority':
            #totwcss = self.nsamples*np.var(self.samples)
            totwcss = np.var(self.samples)
            tovisit.put((0,self.root_node))
        else:
            tovisit.put((None,self.root_node))
        self.leafs = []     
        cur_nodes = set()
        prev_nodes = set()
        prev_nodes.add(self)
        depth = 0
        dsize = 1
        #self.wcss = []
        while not tovisit.empty():
            father_priority,cur = tovisit.get() #it always holds: father_priority = totwcss - wcss(cur) 
            lsamples_idx = []
            rsamples_idx = []
            if cur.nsamples > self.minsamples and (self.visit != 'priority' or dsize < self.nbranchings):
                #km_instance = KMeans(n_clusters=2,n_jobs=-1)
                km_instance = KMeans(n_clusters=2,tol=0.1)
                km = km_instance.fit(self.cluster_matrix[np.array(cur.samples_idx)])
                #self.wcss.append(km.inertia_)
                scaledwcss = km.inertia_/(cur.nsamples*self.dim)
                if self.visit == 'priority' or scaledwcss > self.epsilon: #if km.inertia is still big, we branch on this node
                    for k,label in enumerate(km.labels_):
                        if label == 0:
                            lsamples_idx.append(cur.samples_idx[k]) #independently at which level the node is, we always store absolute indexes for self.samples
                        if label == 1:
                            rsamples_idx.append(cur.samples_idx[k])
                    if self.visit == 'priority':
                        #lwcss = len(lsamples_idx)*np.var([self.samples[k] for k in lsamples_idx])
                        lwcss = np.var([self.samples[k] for k in lsamples_idx])
                        #rwcss = len(rsamples_idx)*np.var([self.samples[k] for k in rsamples_idx])
                        rwcss = np.var([self.samples[k] for k in rsamples_idx])
                        lpriority = totwcss - lwcss
                        rpriority = totwcss - rwcss
                        #print('lwcss = %3.5f \t  rwcss = %3.5f \t  lprio = %3.5f \t  rprio = %3.5f' % (lwcss,rwcss,lpriority,rpriority))
                        dsize += 1
                    else:
                        lpriority = None
                        rpriority = None
                    lnode = Node(lsamples_idx,cur,True)
                    rnode = Node(rsamples_idx,cur,False)
                    cur.children = (lnode,rnode)
                    tovisit.put((lpriority,lnode))
                    tovisit.put((rpriority,rnode))
                    depth = max((depth,lnode.depth,rnode.depth))
            if cur.children is None:
                self.leafs.append(cur)
        self.tree_depth = depth
        self.tree_sparsity = len(self.leafs)/2**self.tree_depth
        
        
class spectral_clustering(Cluster):
    """Clusters data using recursive spectral clustering"""
        
    def __init__(self,samples,epsilon,similarity_measure,simmeasure_beta=0.06,affinity_matrix_threshold=0.5,minsamples=7,implementation='explicit'):
        """	samples: patches to cluster
    		similarity_measure: can be 'frobenius', 'haarpsi' or 'emd' (earth's mover distance)
        	epsilon: threshold for WCSS used as criteria to branch on a tree node
        	affinity_matrix_threshold: threshold in (0,1) for affinity_matrix similarities."""
        
        Cluster.__init__(self,samples)
        self.similarity_measure = similarity_measure
        self.simmeasure_beta = simmeasure_beta
        self.affinity_matrix_threshold = affinity_matrix_threshold
        self.patch_size = self.samples[0].shape
        self.epsilon = epsilon
        self.minsamples = minsamples
        self.implementation = implementation
        self.cluster_matrix = patches2matrix(self.samples).transpose()        

        
    def _cluster(self):
        if self.implementation == 'scikit':
            self._cluster_scikit()
        elif self.implementation == 'explicit':
            self._cluster_explicit()
        self.affinity_matrix_nonzero_perc = len(self.affinity_matrix.data)/len(self.samples)**2

    def _cluster_scikit(self):
        self.root_node = Node(tuple(np.arange(self.nsamples)),None)
        tovisit = []
        tovisit.append(self.root_node)
        self.leafs = []
        cur_nodes = set()
        prev_nodes = set()
        prev_nodes.add(self)
        depth = 0
        if not hasattr(self,'affinity_matrix'):
            self._compute_affinity_matrix()
        def WCSS(clust1_idx,clust2_idx):
            samples1 = [self.samples[i] for i in clust1_idx]
            samples2 = [self.samples[i] for i in clust2_idx]
            cent1 = sum(samples1[1:],samples1[0])
            cent2 = sum(samples2[1:],samples2[0])
            wcss = 0
            for s1,s2 in zip(samples1,samples2):
                wcss += np.linalg.norm(s1-cent1)**2 + np.linalg.norm(s2-cent2)**2
            return(wcss)
        
        while len(tovisit) > 0:
            cur = tovisit.pop()
            lsamples_idx = []
            rsamples_idx = []
            if cur.nsamples > self.minsamples:
                aff_mat = self.affinity_matrix[cur.samples_idx,:][:,cur.samples_idx]
                if (aff_mat == 0).todense().all():
                    continue
                #clust = SpectralClustering(2,affinity='precomputed',eigen_solver='arpack',assign_labels='discretize')
                clust = SpectralClustering(2,affinity='precomputed',eigen_solver='arpack',assign_labels='kmeans')                
                #clust = SpectralClustering(2,affinity='precomputed',eigen_solver='amg',assign_labels='discretize')
                clust.fit(aff_mat)
                for k,label in np.ndenumerate(clust.labels_):
                    k = k[0]
                    if label:
                        lsamples_idx.append(cur.samples_idx[k])
                    else:
                        rsamples_idx.append(cur.samples_idx[k])
                wcss = WCSS(lsamples_idx,rsamples_idx)
                if wcss > self.epsilon:
                    lnode = Node(lsamples_idx,cur,True)
                    rnode = Node(rsamples_idx,cur,False)
                    cur.children = (lnode,rnode)
                    tovisit = [lnode] + tovisit
                    tovisit = [rnode] + tovisit
                    depth = max((depth,lnode.depth,rnode.depth))
                if cur.children is None:
                    self.leafs.append(cur)
        self.tree_depth = depth
        self.tree_sparsity = len(self.leafs)/2**self.tree_depth

    def _cluster_explicit(self):
        self.root_node = Node(tuple(np.arange(self.nsamples)),None)
        tovisit = []
        tovisit.append(self.root_node)
        self.leafs = []
        cur_nodes = set()
        prev_nodes = set()
        prev_nodes.add(self)
        depth = 0
        if not hasattr(self,'affinity_matrix'):
            self._compute_affinity_matrix()
        #np.save('tmpaffmat',self.affinity_matrix)
        #self.affinity_matrix = np.load('tmpaffmat.npy').item()
        self.egvecs = []
                
        def Ncut(D,W,y):
            num = float(y.T.dot(D-W).dot(y))
            den = float(y.T.dot(D).dot(y))
            return(num/den)

        self.idstr = []
        while len(tovisit) > 0:
            cur = tovisit.pop()
            self.idstr.append(cur.idstr)
            lsamples_idx = []
            rsamples_idx = []
            aff_mat = self.affinity_matrix[cur.samples_idx,:][:,cur.samples_idx]
            diag = np.array([row.sum() for row in aff_mat])
            diagsqrt = np.diag(diag**(-1/2))
            laplacian_matrix = diagsqrt.dot(np.diag(diag) - aff_mat).dot(diagsqrt).astype('f')
            #print(depth, cur.nsamples,aff_mat.shape)
            #print('Computing eigenvalues/vectors of %s x %s matrix' % mat.shape)
            egval,egvec = sslinalg.eigsh(laplacian_matrix,k=2,which='SM')
            #print("eigenvalues: ", egval)
            vec = egvec[:,1] #second eigenvalue
            #simple mean thresholding:
            #mean = vec.mean()
            #isinleftcluster = vec > mean
            #isinleftcluster = vec > filters.threshold_otsu(vec)
            isinleftcluster = vec > threshold_otsu(vec)
            self.egvecs.append((cur.depth,vec,isinleftcluster))
            lnonzero,rnonzero = 0,0
            for k,label in np.ndenumerate(isinleftcluster):
                k = k[0]
                if label:
                    lsamples_idx.append(cur.samples_idx[k])
                    if diag[k] != 0:
                        lnonzero += 1
                else:
                    rsamples_idx.append(cur.samples_idx[k])
                    if diag[k] != 0:
                        rnonzero += 1
                    
            #print("left and right cards: ", len(lsamples_idx),len(rsamples_idx))
            try:
                ncutval = Ncut(np.diag(diag),aff_mat,isinleftcluster)
            except ZeroDivisionError:
                continue
            #if np.linalg.norm(aff_mat[1:,1:].todense()) == 0:
            #    ipdb.set_trace()
            #print("Ncut = ", ncutval)]
            #leftaffmatnorm = np.linalg.norm(self.affinity_matrix[lsamples_idx,:][:,lsamples_idx].todense())
            #rightaffmatnorm = np.linalg.norm(self.affinity_matrix[rsamples_idx,:][:,rsamples_idx].todense())
            #if ncutval > self.epsilon and leftaffmatnorm > 0 and rightaffmatnorm > 0:
            #if ncutval > self.epsilon and leftaffmatnorm > 0 and rightaffmatnorm > 0 and cur.nsamples > self.minsamples:
            if ncutval > self.epsilon and len(lsamples_idx) > 0 and len(rsamples_idx) > 0:
                lnode = Node(lsamples_idx,cur,True)
                rnode = Node(rsamples_idx,cur,False)
                cur.children = (lnode,rnode)
                depth = max((depth,lnode.depth,rnode.depth))
                #if len(lsamples_idx) > self.minsamples:
                if lnonzero > self.minsamples:
                    tovisit = [lnode] + tovisit
                else:
                    self.leafs.append(lnode)
                #if len(rsamples_idx) > self.minsamples:
                if rnonzero > self.minsamples:
                    tovisit = [rnode] + tovisit
                else:
                    self.leafs.append(lnode)
            #if cur.children is None:
            else:
                self.leafs.append(cur)
        self.tree_depth = depth
        self.tree_sparsity = len(self.leafs)/2**self.tree_depth

    def plotegvecs(self,savefile=None):
        #self.egvecs.append((cur.depth,vec,isinleftcluster))
        fig,axis = plt.subplots(min(10,len(self.egvecs)),1)
        for idx,ax in np.ndenumerate(axis):
            depth,egv,isinleftcluster = self.egvecs[idx[0]]
            egv.sort()
            isinleftcluster = egv > threshold_otsu(egv)

            t = np.arange(0,len(egv))
            split = isinleftcluster.argmax()
            leftt = t[:split]
            rightt = t[split:]
            ax.plot(leftt,egv[:split],'r-')
            ax.plot(rightt,egv[split:],'b-')
            #plt.plot(0.2*isinleftcluster,'g')
        if savefile is not None:
            plt.savefig(savefile)
        else:
            plt.show()

class fh_union_find():
    """Implementation originally due to Nils Doerrer of union find structure as used by felzenszwalb_huttenlocher_clustering class"""

    def __init__(self, n, k=1):
        """
        Constructor for Segmentation class. Creates a union-find structure
        where all components hold exactly one element. Also initializes
        cardinality and Int lists
        Args:
        n:	number of components in the data structure
        k:	parameter for the threshold function tau = k / |C|
        """
        self.parent = list(range(n))
        self.Int = list(np.zeros(n))
        self.cardinality = list(np.ones(n))
        self.k = k
        self.n = n

    def find(self, i):
        """
        Recursively walks up the tree containing i and gives root node
        Args:
        i:	node to which the root should be found
        Returns:
        root node number (find(root) = root itself)
        """
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j, weight=0):
        """
        Unites two components in the union-find structure. Additionally it
        adjusts the cardinality and Int values such that they are correct for
        the root of each component after merging the components.
        Args:
        i:		node in first component to merge (not neccessarily root)
        j:		node in second component to merge (not neccessarily root)
        weight:	weight of the edge connecting them, needed to set Int value
        """
        i = self.find(i)
        j = self.find(j)
        if i != j:
            self.parent[i] = j
            self.cardinality[j] += self.cardinality[i]
            self.Int[j] = min(self.Int[i], min(self.Int[j], weight))

    def issame(self, i, j):
        """
        Checks whether two nodes are in the same component, i.e. have same root.
        Args:
        i:	first node
        j:	second node
        Returns:
        true iff i and j are in the same component
        """
        return self.find(i) == self.find(j)

    def MInt(self, i, j):
        """
        Returns the MInt value between two components (reasonable iff disjoint)
        MInt = max( Int(C_i) + tau(C_i), Int(C_j) + tau(C_j) )
        Depends on tau function below.
        Args:
        i:	node in the first component
        j:	node in the second component
        Returns:
        MInt(C_i, C_j) value
        """
        i = self.find(i)
        j = self.find(j)
        return max(self.Int[i] - self.tau(i), self.Int[j] - self.tau(j)) # I want MInt to be smaller for low-cardinality sets

    def tau(self, i):
        """
        Represents the threshold function which depends on the component size.
        Here it is set to k / |C| for some constant k.
        Args:
        i: root of a component (to get its cardinality)
        Returns:
        threshold-function value
        """
        return self.k / self.cardinality[i]
        
    def toImage(self, shape):
        """
        Converts the unionfind-structure to an Image of lables.
        For that the shape of the image has to be specified.
        This method only works if the labels correspond to the vertices
        increasing row by row and column by column (sorted Vertex set).
        Args:
        shape:	shape of the original image and also resulting label image
        Returns:
        image (np.ndarray) of the lables in current segmentation state
        """
        labels = np.asarray([self.find(x) for x in range(self.n)]).reshape(shape)
        if DEBUG:
            print("labels")
            print(labels)
            print("cardinality")
            print(np.asarray([self.cardinality[i] if i==self.find(i) else 0 for i in
                              range(self.n)], dtype=np.int32).reshape(shape))
            print("Int")
            print(np.asarray(self.Int).reshape(shape))
            return labels

    def toLabelList(self):
        """
        Converts the unionfind-structure to a list of lists containing
        member indices for each component. Each index is uniquely matched
        to one of the components (at most n).
        Returns:
        list of lists, each holding member indices for each component
        """
        components = []
        result = []
        num = len(self.parent)
        for i in range(num):
            components.append([])
        for i in range(num):
            components[self.find(i)].append(i)
        for entry in components:
            if entry:				# = so if not empty
                result.append(entry)
        return result

            
class felzenszwalb_huttenlocher_clustering(Cluster):
    """Clusters data using adapted Felzenszwalb-Huttenlocher segmentation method"""

    def __init__(self,samples,similarity_measure,simmeasure_beta=0.06,fh_k=1,affinity_matrix_threshold=0.5,minsamples=7):
        Cluster.__init__(self,samples)
        self.similarity_measure = similarity_measure
        self.simmeasure_beta = simmeasure_beta
        self.fh_k = 1
        self.affinity_matrix_threshold = affinity_matrix_threshold
        self.patch_size = self.samples[0].shape
        #self.epsilon = epsilon
        self.minsamples = minsamples
        #self.implementation = implementation
        #self.cluster_matrix = patches2matrix(self.samples).transpose()        
        
    def _cluster(self):
        #E = G.edges(data="weight")
	#E = sorted(E, key=lambda x: x[2])
	#seg = Segmentation(len(V), k=k)
	#for e in E:
	#	(v1, v2, w) = e
	#	i = V.index(v1)
	#	j = V.index(v2)
	#	if w <= seg.MInt(i, j):
	#		seg.union(i, j, weight=w)
	#return(seg)
        if not hasattr(self,'affinity_matrix'):
            self._compute_affinity_matrix(symmetric=False)
        #self.affmat_data,self.affmat_rows,self.affmat_cols are avaiable
        argsort = self.affmat_data.argsort()[::-1]
        edge_weights = self.affmat_data[argsort]
        vertices = list(zip(np.array(self.affmat_rows)[argsort],(np.array(self.affmat_cols)[argsort])))
        self.uf = fh_union_find(len(self.samples), k=self.fh_k)
        nclusters = len(self.samples)
        for ew,v in zip(edge_weights,vertices):
            #print(e,v)
            v1,v2 = v
            #if ew >= self.uf.MInt(v1,v2):
            #    self.uf.union(v1,v2,weight=ew)
            self.uf.union(v1,v2,weight=ew)
            nclusters -= 1
            if nclusters == 2:
                break
            
        
        
class hierarchical_dict(Oc_Dict):

    def __init__(self,clustering,patch_list,dicttype):
        self.clustering = clustering
        if self.clustering.tree_depth == 0:
            raise Exception('Tree depth is 0')
        self.patches = patch_list
        self.patch_size = self.patches[0].shape
        self.npatches = len(patch_list)
        self.dicttype = dicttype
        self.compute()
        Oc_Dict.__init__(self,self.matrix)

    def _compute(self,normalize=True):
        self._compute_centroids(normalize)
        self._compute_haar(normalize)
        self.set_dicttype('haar') #set default dicttype


    def set_dicttype(self, dtype):
        self.dicttype = dtype
        if self.dicttype == 'haar':
            #self.matrix = self.haar_matrix
            #self.normalization_coefficients = self.haar_normalization_coefficients
            self.dictelements = self.haar_dictelements
        elif self.dicttype == 'centroids':
            #self.matrix = self.centroid_matrix
            #self.normalization_coefficients = self.centroid_normalization_coefficients
            self.dictelements = self.centroid_dictelements
        else:
            raise Exception("dicttype must be either 'haar' or 'centroids'")
        self._dictelements_to_matrix()
        self.cardinality = len(self.dictelements)
        
    def _compute_centroids(self, normalize=True):
        leafs = self.clustering.leafs
        ga = (1/self.npatches)*sum(self.patches[1:],self.patches[0])
        if normalize:
            ga /= np.linalg.norm(ga)
        dictelements = [ga] #global average
        for l in leafs:
            centroid = sum([self.patches[i] for i in l.samples_idx[1:]],self.patches[l.samples_idx[0]])
            dictelements += [centroid]
        self.centroid_dictelements = dictelements
        self.centroid_matrix = np.vstack([x.flatten() for x in self.centroid_dictelements]).transpose()
        self.centroid_matrix,self.centroid_normalization_coefficients = normalize_matrix(self.centroid_matrix)
        

    def _compute_haar(self,normalize=True):
        root_node = self.clustering.root_node
        tovisit = [root_node]
        ga = (1/self.npatches)*sum(self.patches[1:],self.patches[0])
        if normalize:
            ga /= np.linalg.norm(ga)
        dictelements = [ga] #global average
        self.leafs = []
        cur_nodes = set()
        prev_nodes = set()
        prev_nodes.add(self)
        depth = 0
        while len(tovisit) > 0:
            cur = tovisit.pop()
            lnode,rnode = cur.children
            lpatches_idx = lnode.samples_idx
            rpatches_idx = rnode.samples_idx
            centroid1 = sum([self.patches[i] for i in lpatches_idx[1:]],self.patches[lpatches_idx[0]])
            centroid2 = sum([self.patches[i] for i in rpatches_idx[1:]],self.patches[rpatches_idx[0]])
            centroid1 /= len(lpatches_idx)
            centroid2 /= len(rpatches_idx)
            curdict = centroid1 - centroid2
            if normalize:
                norm = np.linalg.norm(curdict)
                if norm != 0:
                    curdict /= norm
            if np.isnan(curdict).any():
                ipdb.set_trace()
            dictelements += [curdict]
            if lnode.children is not None:
                tovisit = [lnode] + tovisit
            if rnode.children is not None:
                tovisit = [rnode] + tovisit
        self.haar_dictelements = dictelements
        self.haar_matrix = np.vstack([x.flatten() for x in self.haar_dictelements]).transpose()
        self.haar_matrix,self.haar_normalization_coefficients = normalize_matrix(self.haar_matrix)
        

        
class ksvd_dict(Oc_Dict):
    """Computes dictionary using KSVD method."""
    
    def __init__(self,patch_list,dictsize,sparsity,maxiter=8,implementation='ksvdbox'):
        self.patches = patch_list
        self.npatches = len(patch_list)
        self.patch_size = self.patches[0].shape
        self.dictsize = dictsize
        self.sparsity = sparsity
        self.maxiter = maxiter
        self.implementation = implementation
        from oct2py import octave
        self.octave = octave
        self.octave.addpath('ksvdbox/')
        self.octave.addpath('ompbox/')
        self.octave.addpath(implementation+'/')
        self.compute()
        self.useksvdencoding = True
        Oc_Dict.__init__(self,self.matrix)

    def _ksvd(self):
        """ Requires KSVD.m file"""
        #from oct2py import octave
        param = {'InitializationMethod': 'DataElements',
                 'K': self.dictsize,
                 'L': self.sparsity,
                 'displayProgress': 1,
                 'errorFlag': 0,
                 'numIteration': 10,
                 'preserveDCAtom': 1}
        length = self.patches[0].flatten().shape[0]
        #Y = np.hstack([p.flatten().reshape(length,1) for p in self.patches])
        #octave.addpath('ksvd')
        Y = patches2matrix(self.patches)
        #octave.addpath('../ksvd')
        D = self.octave.KSVD(Y,param)
        self.matrix = D
        self.dictelements = []
        rows,cols = self.patches[0].shape
        for j in range(K):
            self.dictelements.append(D[:,j].reshape(rows,cols))
        
    def _ksvdbox(self):
        """Requires ksvdbox matlab package"""
        length = self.patches[0].flatten().shape[0]
        #Y = np.hstack([p.flatten().reshape(length,1) for p in self.patches])
        Y = patches2matrix(self.patches)
        params = {'data': Y.astype('double'),
                 'Tdata': self.sparsity,
                 'dictsize': self.dictsize,
                 'memusage': 'normal'} #'low','normal' or 'high'
        #[D,X] = self.octave.ksvd(params)
        print('Computing ksvd...')
        [D,X] = self.octave.ksvd(params)
        #[D,X] = octave.ksvd(params)
        print('Done...')
        #[D,X] = self.oc.eval('ksvdparams)
        self.encoding = X.todense()
        self.matrix = D
        self.dictelements = []
        rows,cols = self.patches[0].shape
        for j in range(self.dictsize):
            self.dictelements.append(D[:,j].reshape(rows,cols))

    def _compute(self):
        if self.implementation == 'ksvdbox':
            self._ksvdbox()
        elif self.implementation == 'ksvd':
            self._ksvd()
