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
import skimage.filters as filters
import scipy.sparse
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
def diss_haarpsi(scaling=1):
    ret = lambda patch1,patch2: scaling*(1 - haar_psi(255*patch1,255*patch2)[0])
    return(ret)

def diss_euclidean():
    ret = lambda patch1,patch2: np.linalg.norm(patch1 - patch2)
    return(ret)

def diss_emd(patch_size):
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
    ret = lambda patch1,patch2: pyemd.emd(patch1.flatten(),patch2.flatten(),metric_matrix)
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


def learn_dict(paths,npatches=None,patch_size=(8,8),method='2ddict',transform=None,clustering='2means',cluster_epsilon=2,spectral_dissimilarity='haarpsi',ksvddictsize=10,ksvdsparsity=2,twodpca_l=3,twodpca_r=3,wav_lev=3,dict_with_transformed_data=False,wavelet='haar',dicttype='haar'):
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
    spectral_dissimilarity: dissimilarity measure used for spectral clustering. Can be 'euclidean','haarpsi' or 'emd' (earth's mover distance)
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
    
    #TRANSFORM
    if transform == '2dpca':
        transform_instance = twodpca_transform(patches,twodpca_l,twodpca_r)
    elif transform == 'wavelet':
        transform_instance = wavelet_transform(patches,wav_lev,wavelet)
    elif transform == 'wavelet_packet':
        transform_instance = wavelet_packet(patches,wavelet)
    elif transform is None:
        transform_instance = dummy_transform(patches)
    data_to_cluster = transform_instance.compute()
    
    #CLUSTER
    if clustering is None or method != '2ddict':
        cluster_instance = dummy_clustering(data_to_cluster)
    elif clustering == '2means':
        cluster_instance = twomeans_clustering(data_to_cluster,epsilon=cluster_epsilon)
    elif clustering == 'spectral':
        cluster_instance = spectral_clustering(data_to_cluster,epsilon=cluster_epsilon,dissimilarity=spectral_dissimilarity)    
    cluster_instance.compute()

    #BUILD DICT
    patches4dict = patches
    if dict_with_transformed_data:
        patches4dict = data_to_cluster
    if method == '2ddict':
        dictionary = hierarchical_dict(cluster_instance,patches4dict,dicttype)
    elif method == 'ksvd':
        dictionary = ksvd_dict(patches4dict,dictsize=ksvddictsize,sparsity=ksvdsparsity)
    #dictionary.compute()
    return(dictionary)

def reconstruct(oc_dict,imgpath,sparsity=5,transform=None,wav_lev=3,wavelet='haar'):
    clip = False
    psize = oc_dict.patches[0].shape
    spars= sparsity
    img = np_or_img_to_array(imgpath,psize)
    patches = extract_patches(img,psize)

    #TRANSFORM
    if transform == '2dpca':
        transform_instance = twodpca_transform(patches,twodpca_l,twodpca_r)
    elif transform == 'wavelet':
        transform_instance = wavelet_transform(patches,wav_lev,'haar')
    elif transform == 'wavelet_packet':
        transform_instance = wavelet_packet(patches,wavelet)
    elif transform is None:
        transform_instance = dummy_transform(patches)
    data_to_reconstruct = transform_instance.compute()
    
    outpatches = []
    means,coefs = oc_dict.encode_patches(data_to_reconstruct,spars)
    rec_patches = oc_dict.reconstruct_patches(coefs,means)
    
    #reconstructed_patches = [p.reshape(psize) for p in rec_matrix.transpose()]
    reconstructed_patches = transform_instance.reverse(rec_patches)
    reconstructed = assemble_patches(reconstructed_patches,img.shape)
    if clip:
        reconstructed = clip(reconstructed)
    return(reconstructed,coefs)        

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
        
    def compute(self):
        """Computes the transform and returns a list of patches of the transformed data"""
        return(self._transform())

    def reverse(self,patches):
        """Computes the transform and returns a list of patches of the transformed data"""
        return(self._reverse_transform(patches))

class Cluster(Saveable):
    """Hierarchical cluster of data"""

    def __init__(self,samples):
        self.samples = samples
        self.nsamples = len(samples)
        self.root_node = None

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

    def _matrix_from_patch_list(self,normalize=True):
        if type(self).__name__ == 'hierarchical_dict':
            self.centroid_matrix = np.vstack([x.flatten() for x in self.centroid_dictelements]).transpose()
            self.centroid_matrix,self.centroid_normalization_coefficients = normalize_matrix(self.centroid_matrix)
            self.haar_matrix = np.vstack([x.flatten() for x in self.haar_dictelements]).transpose()
            self.haar_matrix,self.haar_normalization_coefficients = normalize_matrix(self.haar_matrix)
            self.set_dicttype(self.dicttype)
        else:
            self.matrix = np.vstack([x.flatten() for x in self.dictelements]).transpose()
            self.matrix,self.normalization_coefficients = normalize_matrix(self.matrix)


    def compute(self):
        #self.compute_dictelements()
        self._compute()
        self._matrix_from_patch_list(self)

    def min_dissimilarities(self,dissimilarity='euclidean',progress=True):
        """Computes self.min_diss, array of minimum dissimilarities from dictionary atoms to input patches"""
        
        if dissimilarity == 'haarpsi':
            diss = diss_haarpsi(1)
        elif dissimilarity == 'euclidean':
            diss = diss_euclidean()
        elif dissimilarity == 'emd':
            diss = diss_emd(self.patch_size)
        min_diss = []
        counter = 0
        if progress:
            print('Computing minimum dissimilarities with dissimilarity measure: %s' % (dissimilarity))
        for d in self.dictelements:
            md = diss(d,self.patches[0])
            for p in self.patches:
                if progress:
                    print('\r%d/%d' % (counter + 1,len(self.patches)*self.cardinality), sep=' ',end='',flush=True)
                md = min(md,diss(d,p))
                counter += 1
            min_diss.append(md)
        self.min_diss = min_diss
    
    def mutual_coherence(self,progress=True):
        self.max_cor = 0
        self.argmax_cor = None
        ncols = self.matrix.shape[1]
        if progress:
            print('Computing mutual coherence of dictionary')
        counter = 0
        for j1,j2 in itertools.combinations(range(ncols),2):
            if progress:
                print('\r%d/%d' % (counter + 1,ncols*(ncols-1)/2), sep=' ',end='',flush=True)
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
        if hasattr(self,'_encode'):
            self._encode(samples)
        else:
            means = []
            for s in samples.transpose():
                mean = s.mean()
                s -= mean
                means.append(mean)
                #omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity,fit_intercept=True)
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
            #if normalize:
            #    #sample /= outnorm
            #    matrix = self.normalized_matrix
            #    norm_coefs = self.normalization_coefficients
            omp.fit(self.matrix,samples)
        return(means,omp.coef_.transpose())

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
        probs = atoms_prob(coefs)
        maxidx = probs.argsort()[::-1]
        l = int(np.sqrt(natoms))
        rows,cols = (min(10,l),min(10,l))
        patches = []
        for i in range(natoms):
            patches += [self.patches[maxidx[i]]]
        fig, axis = plt.subplots(rows,cols,sharex=True,sharey=True,squeeze=True)
        #for i,j in np.ndindex(rows,cols):
        for idx,ax in np.ndenumerate(axis):
            try:
                i = idx[0]
            except IndexError:
                i = 0
            try:
                j = idx[1]
            except IndexError:
                j = 0                
            ax.set_axis_off()
            ax.imshow(patches[cols*i + j],interpolation='nearest')
        if savefile is None:
            plt.show()
        else:
            plt.savefig(savefile)

            
    def show_dict_patches(self,shape=None,patch_shape=None,savefile=None):
        if patch_shape is None:
            s = int(np.sqrt(self.atom_dim))
            patch_shape = (s,s)
        if shape is None:
            l = int(np.sqrt(self.cardinality))
            shape = (min(10,l),min(10,l))
        rows,cols = shape
        if not hasattr(self,'atom_patches'):
            self.atom_patches = [col.reshape(patch_shape) for col in self.matrix.transpose()]
        self.atoms_by_var = sorted(self.atom_patches,key=lambda x: x.var(),reverse=False)[:rows*cols]
        fig, axis = plt.subplots(rows,cols,sharex=True,sharey=True,squeeze=True)
        #for i,j in np.ndindex(rows,cols):
        for idx,ax in np.ndenumerate(axis):
            try:
                i = idx[0]
            except IndexError:
                i = 0
            try:
                j = idx[1]
            except IndexError:
                j = 0                
            ax.set_axis_off()
            ax.imshow(self.atom_patches[cols*i + j],interpolation='nearest')
        if savefile is None:
            plt.show()
        else:
            plt.savefig(savefile)

class dummy_transform(Transform):
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

    def _transform(self):
        self.transformed_patches = tuple([self._compute_feature_matrix(p) \
                                          for p in self.patch_list])
        return(self.transformed_patches)
        
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
        return(outp)

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
        return(outp)
    
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

    def __init__(self,samples,epsilon,minsamples=5):
        Cluster.__init__(self,samples)
        self.epsilon = epsilon
        self.minsamples = minsamples
        self.cluster_matrix = patches2matrix(self.samples).transpose()

    #@profile
    def _cluster(self):
        self.root_node = Node(tuple(np.arange(self.nsamples)),None)
        tovisit = []
        tovisit.append(self.root_node)
        self.leafs = []
        cur_nodes = set()
        prev_nodes = set()
        prev_nodes.add(self)
        depth = 0
        self.wcss = []
        while len(tovisit) > 0:
            cur = tovisit.pop()
            lsamples_idx = []
            rsamples_idx = []
            if cur.nsamples > self.minsamples:
                #km_instance = KMeans(n_clusters=2,n_jobs=-1)
                km_instance = KMeans(n_clusters=2,tol=0.1)
                km = km_instance.fit(self.cluster_matrix[np.array(cur.samples_idx)])
                self.wcss.append(km.inertia_)
                if km.inertia_ > self.epsilon: #if km.inertia is still big, we branch on this node
                    for k,label in enumerate(km.labels_):
                        if label == 0:
                            lsamples_idx.append(cur.samples_idx[k])
                        if label == 1:
                            rsamples_idx.append(cur.samples_idx[k])
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

class spectral_clustering(Cluster):
    """Clusters data using recursive spectral clustering"""
        
    def __init__(self,samples,epsilon,dissimilarity,affinity_matrix_threshold_perc=0.4,minsamples=7,implementation='explicit'):
        """	samples: patches to cluster
    		dissimilarity: can be 'euclidean', 'haarpsi' or 'emd' (earth's mover distance)
        	epsilon: threshold for WCSS used as criteria to branch on a tree node
        	affinity_matrix_threshold_perc: percentage of nonzero elements in affinity matrix"""
        
        Cluster.__init__(self,samples)
        self.dissimilarity = dissimilarity
        self.affinity_matrix_threshold_perc = affinity_matrix_threshold_perc
        self.patch_size = self.samples[0].shape
        self.epsilon = epsilon
        self.minsamples = minsamples
        self.implementation = implementation
        self.cluster_matrix = patches2matrix(self.samples).transpose()        

    def _compute_diss_rows_cols(self, diss):
        data = []
        rows = []
        cols = []
        counter = 0
        for i,j in itertools.combinations(range(self.nsamples),2):
            print('\r%d/%d' % (counter + 1,self.nsamples*(self.nsamples-1)/2), sep=' ',end='',flush=True)
            d = diss(self.samples[i], self.samples[j])
            data.append(d)
            rows.append(i)
            cols.append(j)
            counter += 1
        return(data,rows,cols)        
    
    def _compute_affinity_matrix(self):
        if self.dissimilarity == 'haarpsi':
            diss = diss_haarpsi(1)
        elif self.dissimilarity == 'euclidean':
            diss = diss_euclidean()
        elif self.dissimilarity == 'emd':
            diss  = diss_emd(self.patch_size)
        print('Computing affinity matrix...')
        data,rows,cols = self._compute_diss_rows_cols(diss)
        thresholded_data = data.copy()
        data.sort()
        thresh = data[int(self.affinity_matrix_threshold_perc*len(data))]
        removed = 0
        for i,d in enumerate(data):
            if d > thresh:
                idx = i-removed
                thresholded_data = thresholded_data[:idx] + thresholded_data[idx+1:]
                rows = rows[:idx] + rows[idx+1:]
                cols = cols[:idx] + cols[idx+1:]
                removed += 1
        data = np.array(thresholded_data)
        self.diss_data = data
        #avgd = data.mean()
        #vard = np.sqrt(data.var())
        #beta = -(np.log(_MIN_SIGNIFICATIVE_MACHINE_NUMBER*_MAX_SIGNIFICATIVE_MACHINE_NUMBER))/(2*(avgd - 5*vard))
        #beta = -(np.log(_MIN_SIGNIFICATIVE_MACHINE_NUMBER))/(avgd - 3*vard)
        #print('\navgd = %f\n vard = %f\n beta = %f' % (avgd,vard,beta))
        data /= data.max()
        beta = 4
        exp_data = np.exp(-beta*data)
        expaff_mat = csr_matrix((exp_data,(rows,cols)),shape=(self.nsamples,self.nsamples))
        print('...done')
        expaff_mat = expaff_mat + expaff_mat.transpose()
        self.affinity_matrix = expaff_mat
        #print(len(self.affinity_matrix.data)/(self.affinity_matrix.shape[0]*self.affinity_matrix.shape[1]))
        
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
        self._compute_affinity_matrix()
        #np.save('tmpaffmat',self.affinity_matrix)
        #self.affinity_matrix = np.load('tmpaffmat.npy').item()
        self.egvecs = []
                
        def Ncut(D,W,y):
            num = float(y.T.dot(D-W).dot(y))
            den = float(y.T.dot(D).dot(y))
            return(num/den)

        while len(tovisit) > 0:
            cur = tovisit.pop()
            lsamples_idx = []
            rsamples_idx = []
            aff_mat = self.affinity_matrix[cur.samples_idx,:][:,cur.samples_idx]
            diag = np.array([row.sum() for row in aff_mat])
            diagsqrt = np.diag(diag**(-1/2))
            mat = diagsqrt.dot(np.diag(diag) - aff_mat).dot(diagsqrt).astype('f')
            #print(depth, cur.nsamples,aff_mat.shape)
            #print('Computing eigenvalues/vectors of %s x %s matrix' % mat.shape)
            egval,egvec = sslinalg.eigsh(mat,k=2,which='SM')
            #print("eigenvalues: ", egval)
            vec = egvec[:,1] #second eigenvalue
            #simple mean thresholding:
            #mean = vec.mean()
            #isinleftcluster = vec > mean
            isinleftcluster = vec > filters.threshold_otsu(vec)
            for k,label in np.ndenumerate(isinleftcluster):
                k = k[0]
                if label:
                    lsamples_idx.append(cur.samples_idx[k])
                else:
                    rsamples_idx.append(cur.samples_idx[k])
            #print("left and right cards: ", len(lsamples_idx),len(rsamples_idx))
            ncutval = Ncut(np.diag(diag),aff_mat,isinleftcluster)
            #if np.linalg.norm(aff_mat[1:,1:].todense()) == 0:
            #    ipdb.set_trace()
            #print("Ncut = ", ncutval)]
            leftaffmatnorm = np.linalg.norm(self.affinity_matrix[lsamples_idx,:][:,lsamples_idx].todense())
            rightaffmatnorm = np.linalg.norm(self.affinity_matrix[rsamples_idx,:][:,rsamples_idx].todense())
            if ncutval > self.epsilon and leftaffmatnorm > 0 and rightaffmatnorm > 0:
                lnode = Node(lsamples_idx,cur,True)
                rnode = Node(rsamples_idx,cur,False)
                cur.children = (lnode,rnode)
                depth = max((depth,lnode.depth,rnode.depth))
                self.egvecs.append((depth,vec,isinleftcluster))
                if len(lsamples_idx) > self.minsamples:
                    tovisit = [lnode] + tovisit
                if len(rsamples_idx) > self.minsamples:
                    tovisit = [rnode] + tovisit
            if cur.children is None:
                self.leafs.append(cur)
        self.tree_depth = depth
        self.tree_sparsity = len(self.leafs)/2**self.tree_depth
        
class hierarchical_dict(Oc_Dict):

    def __init__(self,clustering,patch_list,dicttype):
        self.clustering = clustering
        self.patches = patch_list
        self.patch_size = self.patches[0].shape
        self.npatches = len(patch_list)
        self.dicttype = dicttype
        self.compute()
        Oc_Dict.__init__(self,self.matrix)

    def _compute(self,normalize=True):
        self._compute_centroids(normalize)
        self._compute_haar(normalize)

    def set_dicttype(self, dtype):
        self.dicttype = dtype
        if self.dicttype == 'haar':
            self.matrix = self.haar_matrix
            self.normalization_coefficients = self.haar_normalization_coefficients
            self.dictelements = self.haar_dictelements
        elif self.dicttype == 'centroids':
            self.matrix = self.centroid_matrix
            self.normalization_coefficients = self.centroid_normalization_coefficients
            self.dictelements = self.centroid_dictelements
        else:
            raise Exception("dicttype must be either 'haar' or 'centroids'")
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
        self.octave.addpath('ompbox/')
        self.octave.addpath(implementation+'/')
        self.compute()
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
        params = {'data': Y,
                 'Tdata': self.sparsity,
                 'dictsize': self.dictsize,
                 'memusage': 'normal'} #'low','normal' or 'high'
        #[D,X] = self.octave.ksvd(params)
        print('Computing ksvd...')
        [D,X] = self.octave.ksvd(params)
        print('Done...')
        #[D,X] = self.oc.eval('ksvdparams)
        self.encoding = X
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
