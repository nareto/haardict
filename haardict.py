
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

import ipdb
import itertools
from collections import OrderedDict
import pickle
import matplotlib as mpl
mpl.rc('image', cmap='gray')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import scipy.sparse.linalg as sslinalg
import skimage.io
import skimage.color
from skimage.filters import threshold_otsu
import scipy.sparse
import pyemd
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.cluster import KMeans
from kmaxoids import KMaxoids
from sklearn.cluster import SpectralClustering
from sklearn.cluster import ward_tree
from sklearn.neighbors import kneighbors_graph
from sklearn.feature_extraction.image import extract_patches_2d as extract_patches_w_overlap
from sklearn.feature_extraction.image import reconstruct_from_patches_2d as assemble_patches_w_overlap
import oct2py
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

WAVPACK_CHARS = 'adhv'

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

def np_or_img_to_array(path,crop_to_patchsize=None):
    if path[-3:].upper() in  ['JPG','GIF','PNG','EPS','BMP']:
        ret = skimage.io.imread(path,as_gray=True).astype('float64')/255
    elif path[-3:].upper() in  ['NPY']:
        ret = np.load(path)
    elif path[-4:].upper() == 'TIFF' or path[-3:].upper() == 'CR2':
        ret = read_raw_img(path)
    if crop_to_patchsize is not None:
        m,n = crop_to_patchsize
        M,N = ret.shape
        ret = ret[:M-(M%m),:N-(N%n)]
    return(ret)

def read_raw_img(img,as_gray=True):
    import rawpy
    raw = rawpy.imread(img)
    rgb = raw.postprocess()
    if not as_gray:
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

def sumsupto(k):
    return(k*(k+1)/2)

def orgmode_table_line(strings_or_n):
    out ='| ' + ' | '.join([str(s) for s in strings_or_n]) + ' |'
    return(out)

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

#def is_sorted(values):
#    """Returns True if values is sorted decreasingly, False otherwise"""
#    
#    prev_v = values[0]
#    for v in values[1:]:
#        if v < prev_v:
#            return(False)
#    return(True)

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

def positional_string(encoding):
    """Returns the positional string of a sparse encoding matrix"""
    flat = encoding.transpose().flatten()
    out = np.zeros_like(flat)
    nz = flat.nonzero()[0]
    for idx in nz:
        out[idx] = 1
    return(out)


def extract_patches_wo_overlap(array,size=(8,8)):
    """Returns list of small arrays partitioning the large 2D input array. It's the inverse operation of assemble_patches_wo_overlap"""
    
    ret = []
    height,width = array.shape
    vstep,hstep = size
    for j in range(0,width-hstep+1,hstep):
        for i in range(0,height-vstep+1,vstep):
            subimg = array[i:i+vstep,j:j+hstep].real
            ret.append(subimg)
    return(ret)

def assemble_patches_wo_overlap(patches,out_size):
    """Returns a large 2D array given by row-stacking the arrays in patches, which should be a list. It's the inverse operation of extract_patches"""
    
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


def affinity_matrix(samples,similarity_measure,threshold,symmetric=True):
    """Returns column-sparse representation of matrix of pairwise similarities, keeping only the pairwise similarities that are below the given threshold"""

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
    print('\n %.2f percentage of the affinity matrix entries are non-null' % (100*len(data)/sumsupto(len(samples))))
    if len(rows) < nsamples or len(cols) < nsamples:
        raise Exception("Threshold is set to high: there are isolated vertices")
    data = np.array(data)

    aff_mat = csr_matrix((data,(rows,cols)),shape=(nsamples,nsamples))
    if symmetric:
        aff_mat = aff_mat + aff_mat.transpose() #make it symmetrical
    #print(len(affinity_matrix.data)/(affinity_matrix.shap
    return(aff_mat,data,np.array(rows),np.array(cols))


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



def show_or_save_patches(patchlist,rows,cols,savefile=None):
    """Makes a plot of the patches in patchlist and either shows it or saves it to savefile"""
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
    

class Test(Saveable):
    """Class to test the various dictionaries proposed in the paper for image reconstruction tasks"""

    def __init__(self,learnimgs_paths,npatches=None,patch_size=(8,8),noisevar=0,test_id=None,overlapped_patches=True):
        """
        file_paths: path of image or list of paths of images to extract patches from
        npatches: number of patches to use. If None then all patches will be used for training
        patch_size: size of patches to be extracted
        noisevar: variance of Gaussian noise to be added to input images
        """
        if type(learnimgs_paths) != type([]):
            learnimgs_paths = [learnimgs_paths]
        self.learnimgs_paths = learnimgs_paths
        self.npatches =  npatches
        self.patch_size = patch_size
        self.noisevar = noisevar
        self.debug = False
        self.dictionary = None
        self.reconstructed_img = None
        self.test_results =None
        self.overlapped_patches = overlapped_patches
        if test_id is None:
            now = dt.datetime.now()
            test_id = '-'.join(map(str,[now.year,now.month,now.day])) + '_'+':'.join(map(str,[now.hour,now.minute,now.second]))
        self.test_id = test_id
        self._extract_patches_from_training_imgs()

    def learn_dict(self,method='haar-dict',dictsize=None,clustering='twomeans',cluster_epsilon=None,spectral_similarity='frobenius',simmeasure_beta=0.5,affinity_matrix_threshold=0.5,ksvdsparsity=None,transform=None,twodpca_l=4,twodpca_r=4,wav_lev=1,wavelet='haar'):
        """
        Learns dictionary with selected parameters.

        method: the chosen method. The possible choices are:
            - haar-dict: haar-like coefficients of hierarchical dictionary
            - centroids-dict: centroids of hierarchical dictionary
            - ksvd: uses the KSVD method
            - warmstart: uses 2means-2dict as warm start for KSVD
        dictsize: cardinality of dictionary
        transform: whether to transform the data before applying the method:
            - 2dpca: applies 2DPCA transform (see also: twodpca_l,twodpca_r)
            - wavelet: applies wavelet transform to patches - see also wav_lev, wavelet
            - wavelet_packet: appliest wavelet_packet transform to patches - see also wavelet
        clustering: the clustering procedure (used only for haar-dict and centroids-dict method):
            - twomeans: 2-means on the vectorized samples
            - twomaxoids: 2-maxoids on the vectorized samples
            - spectral: spectral clustering (slow)
            - random: random clustering (used for testing)
        cluster_epsilon: threshold for clustering (lower = finer clustering)
        spectral_similarity: similarity measure used for spectral clustering. Can be 'frobenius','haarpsi' or 'emd' (earth's mover distance)
        simmeasure_beta: beta parameter for Frobenius and Earth Mover's distance similarity measures
        affinity_matrix_threshold: threshold for pairwise similarities in affinity matrix
        ksvdsparsity: sparsity used for OMP in training KSVD. Should be explicitly set.
        twodpca_l,twodpca_r: number of left and right feature vectors used in 2DPCA; the feature matrices will be twodpca_l x twodpca_r
        wavelet: type of wavelet for wavelet or wavelet_packet transformations. Default: haar
        wav_lev: number of levels for the wavelet transform
"""
        self.learning_method = method
        self.learning_transform = transform
        tic()
        self._transform_patches(twodpca_l,twodpca_r,wav_lev,wavelet)
        if self.debug:
            pstr = 'Training dictionary on %d x %d patches with: %s' % (self.patch_size[0],self.patch_size[1],method)
            if method in ['haar-dict', 'centroids-dict']:
                pstr += ' - %s' % clustering
            print(pstr)
        self._train_dict(dictsize,clustering,cluster_epsilon,spectral_similarity,simmeasure_beta,affinity_matrix_threshold,ksvdsparsity)
        self.learning_time = toc(self.debug)


    def reconstruct(self,imgpath,sparsity=2,clip=False):
        """
        Reconstructs image in imgpath with given sparsity using previously learned dictionary 
        """
        self.codeimg_path = imgpath
        self.rec_sparsity = sparsity
        
        if self.dictionary is None:
            raise Exception("No learned dictionary found")
        
        self.codeimg = np_or_img_to_array(imgpath,self.patch_size)
        if self.noisevar == 0:
            img = self.codeimg
            self.noisy_codeimg = None
        else:
            self.noisy_codeimg = self.codeimg + np.random.normal(0,self.noisevar,self.codeimg.shape)
            img = self.noisy_codeimg
        if self.overlapped_patches:
            patches_marray = extract_patches_w_overlap(img,self.patch_size,random_state=None)
            patches = [patches_marray[k,:,:] for k in range(patches_marray.shape[0])]

        else:
            patches = extract_patches_wo_overlap(img,self.patch_size)

        tic()
        outpatches = []
        self.rec_means,self.rec_coefs = self.dictionary.encode_patches(patches,sparsity)
        rec_patches = self.dictionary.reconstruct_patches(self.rec_coefs,self.rec_means)

        #reconstructed_patches = [p.reshape(patch_size) for p in rec_matrix.transpose()]
        if self.overlapped_patches:
            self.reconstructed_img = assemble_patches_w_overlap(np.stack(rec_patches,0),img.shape)
        else:
            self.reconstructed_img = assemble_patches_wo_overlap(rec_patches,img.shape)
        self.reconstruction_time = toc(self.debug)
        if clip:
            self.reconstructed_img = clip(reconstructed)
        self._compute_rec_quality()            
        return(self.reconstructed_img)        


    def _extract_patches_from_training_imgs(self):
        images = []
        for f in self.learnimgs_paths:
            if self.overlapped_patches:
                cleanimg = np_or_img_to_array(f)
            else:
                cleanimg = np_or_img_to_array(f,self.patch_size)                
            if self.noisevar == 0:
                img = cleanimg
            else:
                img = cleanimg + np.random.normal(0,self.noisevar,cleanimg.shape)
            images.append(img)
        patches = []
        for i in images:
            if self.overlapped_patches:
                patches_marray = extract_patches_w_overlap(i,self.patch_size,max_patches=self.npatches,random_state=None)
                patches += [patches_marray[k,:,:] for k in range(patches_marray.shape[0])]
            else:
                patches += [p for p in extract_patches_wo_overlap(i,self.patch_size)]
                #patches = np.array(patches)
                if self.npatches is not None:
                    patches = [patches[i] for i in np.random.permutation(range(len(patches)))][:self.npatches]
        if self.debug:
            print('Extracted %d patches' % len(patches))
        self.training_images = images
        self.patches = patches

    def _transform_patches(self,twodpca_l,twodpca_r,wav_lev,wavelet):
        if self.learning_transform == '2dpca':
            self.transform_instance = twodpca_transform(self.patches,twodpca_l,twodpca_r)
        elif self.learning_transform == 'wavelet':
            self.transform_instance = wavelet_transform(self.patches,wav_lev,wavelet)
        elif self.learning_transform == 'wavelet_packet':
            self.transform_instance = wavelet_packet(self.patches,wavelet)
        elif self.learning_transform is None:
            self.transform_instance = dummy_transform(self.patches)
        self.data_to_cluster = self.transform_instance.transform() #tuple of transformed patches

    def _train_dict(self,dictsize,clustering,cluster_epsilon,spectral_similarity,simmeasure_beta,affinity_matrix_threshold,ksvdsparsity):
        if self.learning_method in ['haar-dict', 'centroids-dict']:
            self.dictionary = hierarchical_dict(self.data_to_cluster)
            self.dictionary.compute(clustering,self.learning_method,nbranchings=dictsize,epsilon=cluster_epsilon,spectral_sim_measure=spectral_similarity,simbeta=simmeasure_beta,affthreshold=affinity_matrix_threshold)
        elif self.learning_method == 'ksvd':
            self.dictionary = ksvd_dict(self.data_to_cluster,dictsize=dictsize,sparsity=ksvdsparsity)
            self.dictionary.compute()
        elif self.learning_method == 'simple-kmeans':
            self.dictionary = simple_clustering_dict(self.data_to_cluster,'Kmeans')
            self.dictionary.compute(dictsize)
        elif self.learning_method == 'simple-kmaxoids':
            self.dictionary = simple_clustering_dict(self.data_to_cluster,'Kmaxoids')
            self.dictionary.compute(dictsize)
        elif self.learning_method == 'warmstart':
            tempdict = hierarchical_dict(self.data_to_cluster)
            tempdict.compute(clustering,nbranchings=dictsize,epsilon=cluster_epsilon,spectral_sim_measure=spectral_similarity,simbeta=simmeasure_beta,affthreshold=affinity_matrix_threshold)
            self.dictionary = ksvd_dict(self.data_to_cluster,dictsize=dictsize,sparsity=ksvdsparsity,warmstart=tempdict)
            self.dictionary.compute()
        #if self.learning_method in ['haar-dict','centroids-dict']:
        #    self.dictionary.haar_dictelements = self.transform_instance.reverse(self.dictionary.haar_dictelements)
        #    self.dictionary.centroid_dictelements = self.transform_instance.reverse(self.dictionary.centroid_dictelements)
        #    self.dictionary.set_dicttype(dicttype)
        else:
            self.dictionary.dictelements = self.transform_instance.reverse(self.dictionary.dictelements)
            self.dictionary._dictelements_to_matrix()

    def _compute_rec_quality(self):
        if self.noisevar > 0:
            n_hpi = haar_psi(255*self.codeimg,255*self.noisy_codeimg)[0]
            n_psnr = psnr(self.codeimg,self.noisy_codeimg)
        else:
            n_hpi = '-'
            n_psnr = '-'
        self.noise_haarpsi = n_hpi
        self.noise_psnr = n_psnr
        self.reconstructed_haarpsi = haar_psi(255*self.codeimg,255*self.reconstructed_img)[0]
        self.reconstructed_psnr = psnr(self.codeimg,self.reconstructed_img)
        self._storage_cost()
        self.qindex =  self.encoding_bits*self.codeimg.size*self.reconstructed_haarpsi/self.storage_cost

    def _storage_cost(self, bits=64):
        """Returns the estimated storage cost of the D,X (dictionary,encoding) pair in bits"""

        self.encoding_bits = bits
        K,N,n = self.dictionary.dictsize, self.dictionary.npatches, self.dictionary.atom_dim
        positional_entropy = entropy(positional_string(self.rec_coefs))
        self.storage_cost = K*n*bits + self.rec_sparsity*N*bits + N*K*positional_entropy

    def show_rec_img(self):
        plt.imshow(self.reconstructed_img,cmap=plt.cm.gray)
        plt.show()

    def _compute_test_results(self):
        if self.reconstructed_img is None:
            raise Exception("No reconstructed image found")
        params = OrderedDict()
        params.update({
            'learning_imgs': self.learnimgs_paths,
            'code_img': self.codeimg_path,
            'patch_size': self.patch_size,
            'overlapped_patches': self.overlapped_patches,
            'n.patches': self.npatches,
            'dictionary_cardinality': self.dictionary.dictsize,
            'learning_method': self.learning_method,
            'transform_on_patches': self.learning_transform
        })
        if self.learning_transform == '2dpca':
            params.update({
                'twodpca_l': self.transform_instance.l,
                'twodpca_r': serf.transform_instance.r
            })
        if self.noisevar is not 0:
            params.update({
                'noisevar': self.noisevar,
                'noisy_img_psnr': self.noise_psnr,
                'noisy_img_haarpsi': self.noise_haarpsi
                })
        if self.learning_method == 'ksvd':
            params.update({
                'ksvd_maxiter': self.dictionary.maxiter,
                'ksvd_training_sparsity': self.dictionary.sparsity 
                })
        elif self.learning_method in ['haar-dict','centroids-dict']:
            params.update({
                'clustering': self.dictionary.clustering_method
                })
            if self.dictionary.clustering_method == 'spectral':
                params.update({
                    'similarity_measure': self.dictionary.clustering.similarity_measure,
                    'affinity_matrix_sparsity': self.dictionary.clustering.affinity_matrix_nonzero_perc
                })
            params.update({
                'tree_visit_type': self.dictionary.visit,
                'tree_depth': self.dictionary.tree_depth,
                'tree_sparsity': self.dictionary.tree_sparsity
                })
        params.update({
            'reconstruction_sparsity': self.rec_sparsity,
            'learning_time': self.learning_time,
            'reconstruction_time': self.reconstruction_time,
            'haarpsi': self.reconstructed_haarpsi,
            'psnr': self.reconstructed_psnr
        })
        self.test_results = params
        
    def print_results(self):
        """
        Prints all the parameters and results of the test
        """
        if self.test_results is None:
            self._compute_test_results()
        print('\n'+10*'-'+'Test results -- ' + self.test_id+10*'-')
        for k,v in self.test_results.items():
            #print(k+': ', v)
            print('%-25s %-25s' % (k, v))

    def print_and_save_orgmode(self,save_prefix,tag=None,simhist=False,saveimgs=False):
        print('\n'+'-'*16+'orgmode output'+'-'*16)
        if tag is None:
            tag = self.test_id
            #tag = ''
        base_save_name = save_prefix + '%s-%s' % (tag,self.learning_method)
        base_save_name += '-%dx%d' % self.patch_size
        if self.learning_method in ['haar-dict','centroids-dict']:
            clust = self.dictionary.clustering_method
            base_save_name += '-'+ clust
            if clust == 'spectral':
                base_save_name += '-' + self.dictionary.clustering.similarity_measure

        tag = 'testid:' + tag
        if simhist:
            nb = int(len(self.dictionary.patches)/5)
            plt.hist(self.dictionary.clustering.affinity_matrix.data,bins=nb)
            simhistpath = base_save_name+'-similarities.png'
            if saveimgs:
                plt.savefig(simhistpath)
            plt.close()
            orgmode_str= '**** similarity histograms\n[[file:%s]]\n' % (simhistpath)
            print(orgmode_str)
        if self.noisevar > 0:
            nimg = base_save_name+'-noisy_image.png'
            plt.imshow(self.noisy_codeimg)
            if saveimgs:
                plt.savefig(nimg)
            plt.close()
            orgmode_str = '\n**** %s noisy image\n[[file:%s]]\n' % (tag, nimg)
            print(orgmode_str)
        recimg = base_save_name+'-reconstructed_image.png'
        plt.imshow(self.reconstructed_img)
        if saveimgs:
            plt.savefig(recimg)
        plt.close()
        orgmode_str = '\n**** %s reconstructed image\n[[file:%s]]\n' % (tag,recimg)
        print(orgmode_str)
        dictelements = base_save_name+'-showdict.png'
        orgmode_str = '**** %s showdict\n[[file:%s]]\n' % (tag,dictelements)
        if saveimgs:
            self.dictionary.show_dict_patches(savefile=dictelements)
        print(orgmode_str)
        plt.close()

        mostused = min(self.dictionary.dictsize,50)
        mostusedatoms = base_save_name+'-mostused.png'
        orgmode_str = '**** %s %d most used atoms\n[[file:%s]]' % (tag,mostused,mostusedatoms)
        if saveimgs:
            self.dictionary.show_most_used_atoms(self.rec_coefs,mostused,savefile=mostusedatoms)
        print(orgmode_str)
        plt.close()

        atomsprob1 = base_save_name+'-atoms_prob.png'
        orgmode_str = '**** %s atoms prob\n[[file:%s]]\n' % (tag, atomsprob1)
        atomsprob2 = base_save_name+'-atoms_prob-sorted.png'
        orgmode_str += '[[file:%s]]'% (atomsprob2)
        pr = atoms_prob(self.rec_coefs)
        plt.plot(pr)
        if saveimgs:
            plt.savefig(atomsprob1)
        plt.close()
        pr.sort()
        pr = pr[::-1]
        plt.plot(pr)
        if saveimgs:
            plt.savefig(atomsprob2)
        plt.close()
        print(orgmode_str)

        if self.dictionary.npatches is not None and self.dictionary.npatches < 100:
            orgmode_str= '**** %s min sim histograms\n' % tag
            print(orgmode_str)
            for sim in ['frobenius','haarpsi','emd']:
                fig = plt.figure()
                self.dictionary.max_similarities(sim,False)
                plt.hist(self.dictionary.max_sim)
                simhist = base_save_name+'-maxsim_%s'%(sim)+'.png'
                orgmode_str = '[[file:%s]]' % simhist
                if saveimgs:
                    fig.savefig(simhist)
                fig.clear()
                plt.close()
                print(orgmode_str)    
        
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
        self.data_dim = self.samples[0].size
        self.nsamples = len(samples)
        self.patch_size = self.samples[0].shape

    def _compute_affinity_matrix(self,**args):
        if self.similarity_measure == 'haarpsi':
            self.simmeasure = simmeasure_haarpsi()
        elif self.similarity_measure == 'frobenius':
            self.simmeasure = simmeasure_frobenius(beta=self.simmeasure_beta,samples=self.samples)
        elif self.similarity_measure == 'emd':
            self.simmeasure  = simmeasure_emd(self.patch_size,beta=self.simmeasure_beta,samples=self.samples)
        self.affinity_matrix,self.affmat_data,self.affmat_rows,self.affmat_cols = affinity_matrix(self.samples,self.simmeasure,self.affinity_matrix_threshold,args)
            #print(len(self.affinity_matrix.data)/(self.affinity_matrix.shape[0]*self.affinity_matrix.shape[1]))
        self.affinity_matrix_nonzero_perc = len(self.affinity_matrix.data)/self.nsamples**2

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
        
    
class Node():
    def __init__(self,samples_idx,parent,isleftchild=True,representative_patch=None):
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
        self.samples_idx = np.array(samples_idx) #indexes of d.samples_idx where d is an ocdict
        self.nsamples= len(self.samples_idx)
        self.representative_patch = representative_patch
        #if samples_idx is not None:
        #    self.samples_idx = tuple(samples_idx) #indexes of d.samples_idx where d is an ocdict
        #    self.nsamples= len(self.samples_idx)

    #def samples_list(self,ocdict):
    #    return([ocdict.samples[i] for i in self.samples_idx])



class Oc_Dict(Saveable):
    """Dictionary"""

    def __init__(self,patch_list):
        self.patches = np.array(patch_list)
        #self.matrix = matrix
        #self.shape = matrix.shape
        #self.atom_dim,self.cardinality = matrix.shape
        #self.patches_dim = patches_dim
        #if patches_dim is not None:
        #    self.twod_dict = True
        self.atom_dim = self.patches[0].size
        self.patch_size = self.patches[0].shape
        self.npatches = len(patch_list)


    #def _compute(self):
    #    #self.compute_dictelements()
    #    self._dictelements_to_matrix()
        
    def _dictelements_to_matrix(self):
        self.matrix = np.vstack([x.flatten() for x in self.dictelements]).transpose()
        self.matrix,self.normalization_coefficients = normalize_matrix(self.matrix)
        self.atom_dim = self.matrix.shape[0]

        
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
                    print('\r%d/%d' % (counter + 1,len(self.patches)*self.dictsize), sep=' ',end='',flush=True)
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

    def encode_samples(self,samples,sparsity):
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

    def encode_patches(self,patches,sparsity):
        return(self.encode_samples(patches2matrix(patches),sparsity))
        
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
            l = int(np.sqrt(self.dictsize))
            shape = (min(10,l),min(10,l))
        rows,cols = shape
        if self.dictsize < 15:
            rows,cols = (1,self.dictsize)
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
        return(np.dot(np.dot(self.V.transpose(),patch),self.U).real)

    def _invert_feature_matrix(self,fmat):
        return(np.dot(np.dot(self.V,fmat),self.U.transpose()).real)

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
    
    def cluster(self,patch_indexes):
        ran = np.random.randint(0,2,len(patch_indexes))
        lsamples = (ran == 0).nonzero()[0]
        rsamples = (ran != 0).nonzero()[0]
        lcentroid = None
        rcentroid = None
        curpatches = self.samples[patch_indexes]
        if len(lsamples) > 0 and len(rsamples) > 0:
            lcentroid = sum([curpatches[i] for i in lsamples[1:]],curpatches[lsamples[0]])
            rcentroid = sum([curpatches[i] for i in rsamples[1:]],curpatches[rsamples[0]])
            lcentroid /= len(lsamples)
            rcentroid /= len(rsamples)
        
        return(lsamples,rsamples,None)
    
class twomeans_clustering(Cluster):
    """Clusters data using recursive 2-means"""

    def __init__(self,patches):
        Cluster.__init__(self,patches)
        self.cluster_matrix = patches2matrix(self.samples).transpose()

    def cluster(self, patch_indexes):
        """Returns (idx1,idx2,wcss) where idx1 and idx2 are the set of indexes partitioning the data array and wcss is the achieved value of the WCSS function"""

        km_instance = KMeans(n_clusters=2,tol=0.1)
        km = km_instance.fit(self.cluster_matrix[patch_indexes,:])
        #self.wcss.append(km.inertia_)
        lsamples = (km.labels_ == 0).nonzero()[0]
        rsamples = (km.labels_ != 0).nonzero()[0]
        scaledwcss = km.inertia_/(len(patch_indexes*self.data_dim))

        lcentroid = None
        rcentroid = None
        curpatches = self.samples[patch_indexes]
        if len(lsamples) > 0 and len(rsamples) > 0:
            lcentroid = sum([curpatches[i] for i in lsamples[1:]],curpatches[lsamples[0]])
            rcentroid = sum([curpatches[i] for i in rsamples[1:]],curpatches[rsamples[0]])
            lcentroid /= len(lsamples)
            rcentroid /= len(rsamples)
        return(lsamples,rsamples,lcentroid,rcentroid,scaledwcss)
        
class twomaxoids_clustering(Cluster):
    """Clusters data using recursive 2-maxoids"""

    def __init__(self,patches):
        Cluster.__init__(self,patches)
        self.cluster_matrix = patches2matrix(self.samples)

    def cluster(self, patch_indexes,maxoids_as_representative=True):
        """Returns (idx1,idx2,wcss) where idx1 and idx2 are the set of indexes partitioning the data array and wcss is the achieved value of the WCSS function"""
        
        kmax = KMaxoids(self.cluster_matrix[:,patch_indexes],K=2)
        maxoids,labels = kmax.run()
        #lsamples = []
        #for i in clusters[0]:
        #    lsamples.append(i)
        #rsamples = []
        #for i in clusters[1]:
        #    rsamples.append(i)
        #lsamples = np.array(lsamples)
        #rsamples = np.array(rsamples)
        lsamples = (labels == 0).nonzero()[0]
        rsamples = (labels != 0).nonzero()[0]
        scaled_val = kmax.val/(len(patch_indexes*self.data_dim))
        pshape = self.samples[0].shape
        if maxoids_as_representative:
            lmax = maxoids[:,0].reshape(pshape)
            rmax = maxoids[:,1].reshape(pshape)
            return(lsamples,rsamples,lmax,rmax,scaled_val)
        else:
            lcentroid = None
            rcentroid = None
            curpatches = self.samples[patch_indexes]
            if len(lsamples) > 0 and len(rsamples) > 0:
                lcentroid = sum([curpatches[i] for i in lsamples[1:]],curpatches[lsamples[0]])
                rcentroid = sum([curpatches[i] for i in rsamples[1:]],curpatches[rsamples[0]])
                lcentroid /= len(lsamples)
                rcentroid /= len(rsamples)
            return(lsamples,rsamples,lcentroid,rcentroid,scaled_val)
        

        

    
class spectral_clustering(Cluster):
    """Clusters data using recursive spectral clustering"""
        
    def __init__(self,samples,similarity_measure,simmeasure_beta=0.06,affinity_matrix_threshold=0.5):
        """	samples: patches to cluster
    		similarity_measure: can be 'frobenius', 'haarpsi' or 'emd' (earth's mover distance)
        	affinity_matrix_threshold: threshold in (0,1) for affinity_matrix similarities."""
        
        Cluster.__init__(self,samples)
        self.similarity_measure = similarity_measure
        self.simmeasure_beta = simmeasure_beta
        self.affinity_matrix_threshold = affinity_matrix_threshold
    
        self.cluster_matrix = patches2matrix(self.samples).transpose()        
        self._compute_affinity_matrix()
        
    def cluster(self, patch_indexes):
        return(self._cluster_scikit(patch_indexes))
        #return(self._cluster_explicit(patch_indexes))

    def _Ncut(self,W,y,D=None):
        if D is None:
            diag = np.array([row.sum() for row in W])
            D = scipy.sparse.diags(diag)
        #num = float(y.T.dot(D-W).dot(y))  #this doesn't work because of np.dot not being aware of sparse matrices. so horrible hacks follow
        #den = float(y.T.dot(D).dot(y))
        y = np.hstack(y)
        #lterm = np.hstack(scipy.sparse.csr_matrix.dot(y.T,D-W))
        lterm = scipy.sparse.csr_matrix.dot(y.T,D-W)
        rterm = y.T
        num = np.inner(lterm,rterm)
        den = np.inner(np.inner(y,D.diagonal()),y)
        return(num/den)

        
    def _cluster_scikit(self, patch_indexes):
        """Returns (idx1,idx2,ncut) where idx1 and idx2 are the set of indexes partitioning the data array and ncut is the achieved value of the NCut function. Scikit's implementation of spectral clustering is used."""

        aff_mat = self.affinity_matrix[patch_indexes,:][:,patch_indexes]
        #if (aff_mat == 0).todense().all():
        #    continue
        #sc = SpectralClustering(2,affinity='precomputed',eigen_solver='arpack',assign_labels='discretize')
        sc = SpectralClustering(2,affinity='precomputed',eigen_solver='arpack',assign_labels='kmeans')                
        #sc = SpectralClustering(2,affinity='precomputed',eigen_solver='amg',assign_labels='discretize')
        sc.fit(aff_mat)
        lsamples = (sc.labels_ == 0).nonzero()[0]
        rsamples = (sc.labels_ != 0).nonzero()[0]
        ncut = self._Ncut(aff_mat,sc.labels_)

        lcentroid = None
        rcentroid = None
        curpatches = self.samples[patch_indexes]
        if len(lsamples) > 0 and len(rsamples) > 0:
            lcentroid = sum([curpatches[i] for i in lsamples[1:]],curpatches[lsamples[0]])
            rcentroid = sum([curpatches[i] for i in rsamples[1:]],curpatches[rsamples[0]])
            lcentroid /= len(lsamples)
            rcentroid /= len(rsamples)

        return(lsamples,rsamples,lcentroid,rcentroid,ncut,None)
        

    def _cluster_explicit(self,patch_indexes):
        """Returns (idx1,idx2,ncut,egvec) where idx1 and idx2 are the set of indexes partitioning the data array, ncut is the achieved value of the NCut function and egvec is the eigenvector achieving this value. Eigenvalues/vectors of the Laplacian are explicitly computed."""
        
        aff_mat = self.affinity_matrix[patch_indexes,:][:,patch_indexes]
        diag = np.array([row.sum() for row in aff_mat])
        D = scipy.sparse.diags(diag)
        #diagsqrt = np.diag(diag**(-1/2))
        diagsqrt = scipy.sparse.diags(np.hstack(D.data**(-1/2)))
        laplacian_matrix = diagsqrt.dot(D - aff_mat).dot(diagsqrt).astype('f')
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
        lnonzero,rnonzero = 0,0
        lsamples = (isinleftcluster == 0).nonzero()[0]
        rsamples = (isinleftcluster != 0).nonzero()[0]
        #try:
        ncutval = self._Ncut(aff_mat,isinleftcluster.reshape(len(patch_indexes),1),D)
        #except ZeroDivisionError:
        #    continue

        lcentroid = None
        rcentroid = None
        curpatches = self.samples[patch_indexes]
        if len(lsamples) > 0 and len(rsamples) > 0:
            lcentroid = sum([curpatches[i] for i in lsamples[1:]],curpatches[lsamples[0]])
            rcentroid = sum([curpatches[i] for i in rsamples[1:]],curpatches[rsamples[0]])
            lcentroid /= len(lsamples)
            rcentroid /= len(rsamples)

        return(lsamples,rsamples,lcentroid,rcentroid,ncutval,vec)

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


class simple_clustering_dict(Oc_Dict):

    def __init__(self,patch_list,clustering='Kmeans'):
        Oc_Dict.__init__(self,patch_list)
        self.clustering = clustering

    def compute(self,dict_card):
        data_mat = patches2matrix(self.patches)
        if self.clustering == 'Kmeans':
            km = KMeans(n_clusters=dict_card,tol=0.1).fit(data_mat.transpose())
            self.matrix = km.cluster_centers_.transpose()
        elif self.clustering == 'Kmaxoids':
            kM = KMaxoids(data_mat,K=dict_card)
            self.matrix,labels = kM.run()
        self.dictelements = matrix2patches(self.matrix)
        self.dictsize = dict_card
            
        
class hierarchical_dict(Oc_Dict):

    def __init__(self,patch_list):
        Oc_Dict.__init__(self,patch_list)
        self.dicttype = 'haar'

    def compute(self,clustering_method,dicttype='haar-dict',nbranchings=None,epsilon=None,minsamples=5,\
                spectral_sim_measure='frobenius',simbeta=0.06,affthreshold=0.5):
        if (nbranchings is None and epsilon is None) or (nbranchings is not None and epsilon is not None):
            raise Exception('Exactly one of nbranchings or epsilon has to be set')
        self.clustering_method = clustering_method
        self.dicttype = dicttype
        if clustering_method == 'twomeans':
            self.clustering = twomeans_clustering(self.patches)
        elif clustering_method == 'twomaxoids':
            self.clustering = twomaxoids_clustering(self.patches)
        elif clustering_method == 'spectral':
            self.clustering = spectral_clustering(self.patches,similarity_measure=spectral_sim_measure,\
                                                  simmeasure_beta=simbeta,affinity_matrix_threshold=affthreshold)
        elif clustering_method == 'random':
            self.clustering = monkey_clustering(self.patches)
        else:
            raise Exception('clustering_method must be either \'twomeans\',\'spectral\' or \'random\'')
        self.root_node = Node(tuple(np.arange(self.npatches)),None)
        self.clust_epsilon = epsilon
        if nbranchings is not None:
            self.visit = 'priority'
            tovisit = queue.PriorityQueue()
        else:
            self.visit = 'fifo'
            tovisit = queue.Queue()
            #tovisit = queue.LifoQueue()
        if self.visit == 'priority':
            #totwcss = self.nsamples*np.var(self.samples)
            totvar = np.var(np.array(self.patches))
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
            if cur.nsamples > minsamples and (self.visit != 'priority' or dsize < nbranchings):
                if clustering_method == 'spectral':
                    curlsamples, currsamples, lrepr, rrepr, clust_ret_val,egvec = self.clustering.cluster(cur.samples_idx)
                else:
                    curlsamples, currsamples, lrepr, rrepr, clust_ret_val = self.clustering.cluster(cur.samples_idx)
                #has_sons = True if curlsamples is not None and currsamples is not None else False
                #if has_sons and (self.visit == 'priority' or clust_ret_val > self.clust_epsilon): #decide whether or not to branch
                minsize = min(len(curlsamples),len(currsamples))
                if minsize > 0 and (self.visit == 'priority' or clust_ret_val > self.clust_epsilon): #decide whether or not to branch
                    abs_lsamples = cur.samples_idx[curlsamples]
                    abs_rsamples = cur.samples_idx[currsamples]
                    if self.visit == 'priority':
                        #lwcss = len(lsamples_idx)*np.var([self.samples[k] for k in lsamples_idx])
                        lvar = np.var(self.patches[abs_lsamples])
                        #rwcss = len(rsamples_idx)*np.var([self.samples[k] for k in rsamples_idx])
                        rvar = np.var(self.patches[abs_rsamples])
                        lpriority = totvar - lvar
                        rpriority = totvar - rvar
                        #print('lwcss = %3.5f \t  rwcss = %3.5f \t  lprio = %3.5f \t  rprio = %3.5f' % (lwcss,rwcss,lpriority,rpriority))
                        dsize += 1
                    else:
                        lpriority = None
                        rpriority = None
                    if lpriority == rpriority:
                        rpriority += np.finfo(float).eps
                    lnode = Node(abs_lsamples,cur,True, lrepr)
                    rnode = Node(abs_rsamples,cur,False, rrepr)
                    cur.children = (lnode,rnode)
                    tovisit.put((lpriority,lnode))
                    tovisit.put((rpriority,rnode))
                    depth = max((depth,lnode.depth,rnode.depth))
            if cur.children is None:
                self.leafs.append(cur)
        self.tree_depth = depth
        self.tree_sparsity = len(self.leafs)/2**self.tree_depth
        self.tree2dict()
        #self._compute()
        
    def tree2dict(self,normalize=True):
        """Visits the tree and recomputes the dictionary elements"""

        if self.tree_depth == 0:
            raise Exception('Tree depth is 0')
        if self.dicttype == 'haar-dict':
            self._tree2dict_haar(normalize)
            self.dictelements = self.haar_dictelements
        elif self.dicttype == 'centroids-dict':
            self._tree2dict_centroids(normalize)
            self.dictelements = self.centroid_dictelements
        self._dictelements_to_matrix()
        self.dictsize = len(self.dictelements)
        #self.set_dicttype(self.dicttype) #set default dicttype

    #def set_dicttype(self, dtype):
    #    self.dicttype = dtype
    #    if self.dicttype == 'haar-dict':
    #        #self.matrix = self.haar_matrix
    #        #self.normalization_coefficients = self.haar_normalization_coefficients
    #        self.dictelements = self.haar_dictelements
    #    elif self.dicttype == 'centroids-dict':
    #        #self.matrix = self.centroid_matrix
    #        #self.normalization_coefficients = self.centroid_normalization_coefficients
    #        self.dictelements = self.centroid_dictelements
    #    else:
    #        raise Exception("dicttype must be either 'haar' or 'centroids'")
    #    self._dictelements_to_matrix()
    #    self.dictsize = len(self.dictelements)
        
    def _tree2dict_centroids(self, normalize=True):
        leafs = self.leafs
        ga = (1/self.npatches)*sum(self.patches[1:],self.patches[0])
        if normalize:
            ga /= np.linalg.norm(ga)
        dictelements = [ga] #global average
        for l in leafs:
            #centroid = sum([self.patches[i] for i in l.samples_idx[1:]],self.patches[l.samples_idx[0]])
            #dictelements += [centroid]
            dictelements += [l.representative_patch]
        self.centroid_dictelements = dictelements
        self.centroid_matrix = np.vstack([x.flatten() for x in self.centroid_dictelements]).transpose()
        self.centroid_matrix,self.centroid_normalization_coefficients = normalize_matrix(self.centroid_matrix)
        

    def _tree2dict_haar(self,normalize=True):
        root_node = self.root_node
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
            #lpatches_idx = lnode.samples_idx
            #rpatches_idx = rnode.samples_idx
            #centroid1 = sum([self.patches[i] for i in lpatches_idx[1:]],self.patches[lpatches_idx[0]])
            #centroid2 = sum([self.patches[i] for i in rpatches_idx[1:]],self.patches[rpatches_idx[0]])
            #centroid1 /= len(lpatches_idx)
            #centroid2 /= len(rpatches_idx)
            #curdict = centroid1 - centroid2
            curdict = lnode.representative_patch - rnode.representative_patch
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
    
    def __init__(self,patch_list,dictsize,sparsity,maxiter=10,warmstart=None):
        Oc_Dict.__init__(self,patch_list)
        self.npatches = len(patch_list)
        self.dictsize = dictsize
        self.sparsity = sparsity
        self.maxiter = maxiter
        from oct2py import octave
        self.octave = octave
        #self.octave.addpath(implementation+'/')
        self.useksvdencoding = False
        self.warmstart = warmstart


    def compute(self):
        self._ksvdbox()
        #self._ksvd()

    def _ksvd(self):
        """ Requires KSVD.m file"""
        
        #from oct2py import octave
        param = {'InitializationMethod': 'DataElements',
                 'K': self.dictsize,
                 'L': self.sparsity,
                 'displayProgress': 1,
                 'errorFlag': 0,
                 'numIteration': self.maxiter,
                 'preserveDCAtom': 1}
        length = self.patches[0].flatten().shape[0]
        #Y = np.hstack([p.flatten().reshape(length,1) for p in self.patches])
        self.octave.addpath('ksvd')
        Y = patches2matrix(self.patches)
        #octave.addpath('../ksvd')
        D = self.octave.KSVD(Y,param)
        self.matrix = D
        self.dictelements = []
        rows,cols = self.patches_shape
        for j in range(K):
            self.dictelements.append(D[:,j].reshape(rows,cols))
        
    def _ksvdbox(self):
        """Requires ksvdbox matlab package"""

        self.octave.addpath('ompbox/')
        self.octave.addpath('ksvdbox/')
        length = self.patches[0].flatten().shape[0]
        #Y = np.hstack([p.flatten().reshape(length,1) for p in self.patches])
        Y = patches2matrix(self.patches)
        if self.warmstart is None:
            params = {'data': Y.astype('double'),
                      'Tdata': self.sparsity,
                      'dictsize': self.dictsize,
                      #'iternum': 10,
                      'memusage': 'normal'} #'low','normal' or 'high'
        else:
            params = {'data': Y.astype('double'),
                      'Tdata': self.sparsity,
                      'initdict': self.warmstart.matrix,
                      'iternum': 7,
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
        rows,cols = self.patch_size
        for j in range(self.dictsize):
            self.dictelements.append(D[:,j].reshape(rows,cols))

