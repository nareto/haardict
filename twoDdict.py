import ipdb
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
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import ward_tree
from sklearn.neighbors import kneighbors_graph
import oct2py
import gc
import queue
import pywt

_METHODS_ = ['2ddict-2means','2ddict-2means-haar','2ddict-spectral','ksvd']

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

def HaarPSI(img1,img2):
    """Computes HaarPSI of img1 vs. img2. Requires file HaarPSI.m to be present in working directory"""
    from oct2py import octave
    img1 = img1.astype('float64')
    img2 = img2.astype('float64')
    octave.eval('pkg load image')
    haarpsi = octave.HaarPSI(255*img1,255*img2)
    return(haarpsi)

def vecHaarPSI(arr1,arr2,shape=(8,8)):
    #from oct2py import octave
    #octave.eval('pkg load image')
    oc = oct2py.Oct2Py()
    haarpsi = oc.HaarPSI(arr1.reshape(shape),arr2.reshape(shape))
    #print(haarpsi)
    #ipdb.set_trace()
    return(haarpsi)

def affinity_matrix(X,sigma=1,threshold=0.5):
    """Computes affinity matrix of the distance-weighted graph built from X's rows"""
    nsamples,dim = X.shape
    W = np.zeros(shape=(nsamples,nsamples))
    dists = []
    for i in range(nsamples):
        for j in range(i+1):
            v_i = X[i,:]
            v_j = X[j,:]
            d_ij = np.linalg.norm(v_i - v_j)
            dists.append(d_ij)
            if d_ij < threshold:
                w_ij = np.exp(-d_ij/sigma)
                W[i,j] = w_ij
                W[j,i] = w_ij
    return(W,dists)


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
    baseexp = int(np.log2(coeffs[0].shape[0]))
    tot_length = int(4**baseexp + sum([3*4**(baseexp+i) for i in range(levels)]))
    out = np.zeros(tot_length)
    out[:4**baseexp] = coeffs[0].flatten()
    offset = 4**baseexp
    for lev,details in enumerate(coeffs[1:]):
        dh,dv,dd = details
        length = 4**(baseexp+lev)
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

def learn_dict(paths,l=3,r=3,method='2ddict-2means',clusteronpatches=False,**other_args):
    """Learns dictionary based on the selected method. 

    paths: list of paths of images to learn the dictionary from
    l,r: number of left and right feature vectors used in 2DPCA; the feature matrices will be l x r
    method: the chosen method. The possible choices are:
    	- 2ddict-2means: 2ddict procedure using 2-means to cluster the feature patches
    	- 2ddict-spectral: 2ddict procedure using spectral clustering on the patches
    	- ksvd: uses the KSVD method
    """
     
    if method not in _METHODS_:
        raise Exception("'method' has to be on of %s" % _METHODS_)
    elif method == 'ksvd' and ('dictsize' not in other_args.keys() or 'sparsity'  not in other_args.keys()):
        raise Exception("KSVD method requires setting variables 'dictsize' and 'sparsity'")
    images = []
    for f in paths:
        if f[-3:].upper() in  ['JPG','GIF','PNG','EPS']:
            images.append(skimage.io.imread(f,as_grey=True))
        elif f[-3:].upper() in  ['NPY']:
            images.append(np.load(f))
    print('Learning from images: %s' % paths)

    patches = []
    for i in images:
        patches += [Patch(p) for p in extract_patches(i)]

        twodpca_instance = None
    if not clusteronpatches:
        twodpca_instance = twodpca(patches,l,r)
        twodpca_instance.compute_simple_bilateral_2dpca()
        for p in patches:
            p.compute_feature_matrix(twodpca_instance.U,twodpca_instance.V)

    ocd = ocdict(patches)
    if method[:6] == '2ddict':
        try:
            cluster_epsilon = other_args['cluster_epsilon']
        except KeyError:
            cluster_epsilon = 1e-2
    if method == '2ddict-2means':
        ocd.twomeans_cluster(clusteronpatches=clusteronpatches,epsilon=cluster_epsilon)
    elif method == '2ddict-2means-haar':
        ocd.twomeans_haar_cluster(epsilon=cluster_epsilon)
    elif method == '2ddict-spectral':
        ocd.spectral_cluster(epsilon=cluster_epsilon,clusteronpatches=clusteronpatches)
    #elif method == '2ddcit-ward':
    #    ocd.ward_cluster()
    elif method == 'ksvd':
        ocd.ksvdbox_dict(dictsize=other_args['dictsize'],sparsity=other_args['sparsity'])
    ocd._compute_matrix()
    ocd._normalize_matrix()
    return(twodpca_instance,ocd)

def reconstruct(ocdict,imgpath,sparsity=5,plot=True,retimg=False):

    clip = False
    twodpca = None

    psize = (ocdict.height,ocdict.width)
    spars= sparsity

    if imgpath[-3:].upper() in  ['JPG','GIF','PNG','EPS']:
        img = skimage.io.imread(imgpath,as_grey=True)
    elif imgpath[-3:].upper() in  ['NPY']:
        img = np.load(imgpath)
    elif imgpath[-4:].upper() == 'TIFF' or imgpath[-3:].upper() == 'CR2':
        img = twoDdict.rescale(twoDdict.read_raw_img(imgpath))
    patches = twoDdict.extract_patches(img,size=psize)
    outpatches = []
    coefsnonzeros = []
    for p in patches:
        patch = twoDdict.Patch(p)
        coefs,mean = ocdict.sparse_code(patch,spars)
        outpatches.append(ocdict.decode(coefs,mean))
        coefsnonzeros.append(len(coefs.nonzero()[0]))

    out = twoDdict.assemble_patches(outpatches,img.shape)
    if clip:
        out = twoDdict.clip(out)
    hpi = twoDdict.HaarPSI(img,out)
    print('HaarPSI = %f ' % hpi)
    psnrval = twoDdict.psnr(img,out)
    print('PSNR = %f  ' % psnrval)
    twonorm = np.linalg.norm(img-out,ord=2)
    print('2 norm = %f' % twonorm)
    fronorm = np.linalg.norm(img-out,ord='fro')
    print('Frobenius norm = %f' % fronorm)
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)#, sharey=True)
        #ax1.imshow(img[34:80,95:137], cmap=plt.cm.gray,interpolation='none')
        #ax2.imshow(out[34:80,95:137], cmap=plt.cm.gray,interpolation='none')
        ax1.imshow(img, cmap=plt.cm.gray,interpolation='none')
        ax2.imshow(out, cmap=plt.cm.gray,interpolation='none')
        #ax3.imshow(outclip, cmap=plt.cm.gray,interpolation='none')
        #fig.show()
        plt.show()
    if retimg:
        return(out)


def allhaar(limg,rimg,ocd=None,cluster_epsilon=2,sparsity=2):

    if limg[-3:].upper() in  ['JPG','GIF','PNG','EPS']:
        img1 = skimage.io.imread(limg,as_grey=True)
    elif limg[-3:].upper() in  ['NPY']:
        img1 = np.load(limg)

    if limg[-3:].upper() in  ['JPG','GIF','PNG','EPS']:
        img2 = skimage.io.imread(limg,as_grey=True)
    elif limg[-3:].upper() in  ['NPY']:
        img2 = np.load(limg)

    if ocd is None:
        print('learning')
        patches1 = [Patch(p) for p in extract_patches(img1)]
        ocd = ocdict(patches1)
        ocd.twomeans_haar_cluster(epsilon=cluster_epsilon)
        print('computing stuff')
        ocd.normalized_matrix = np.zeros((64,ocd.ndictelements))
        for i,d in enumerate(ocd.dictelements):
            vec = pywt2array(pywt.wavedec2(d.matrix,'haar','periodic',3))
            #vec = d.matrix.flatten()
            vec -= vec.mean()
            ocd.normalized_matrix[:,i] = vec/np.linalg.norm(vec)
            #ocd.normalized_matrix[:,i] = vec
    ocd.matrix_computed = True
    
    print('reconstructing')
    npatches = 200
    patches2 = [Patch(p) for p in extract_patches(img2)][:npatches]
    outpatches = []
    coefsnonzeros = []
    for p in patches2:
        p.orig_matrix = p.matrix
        p.matrix = pywt2array(pywt.wavedec2(p.matrix,'haar','periodic',3)).reshape(8,8)
        coefs,mean = ocd.sparse_code(p,sparsity)
        #coefs,mean = ocd.sparse_code_ompext(p,sparsity,ompbox=False)
        #ipdb.set_trace()
        dec = ocd.decode(coefs,mean).flatten()
        #dec = ocd.decode(coefs,mean).flatten().transpose()
        #outpatches.append(dec.reshape(8,8))
        outpatches.append(pywt.waverec2(array2pywt(dec,3),'haar','periodic'))
        #outpatches.append(ocd.decode(coefs,mean))
        coefsnonzeros.append(len(coefs.nonzero()[0]))
    errors = []
    print('computing errors')
    for i in range(npatches):
        inp,outp = patches2[i].orig_matrix,outpatches[i]
        rout = rescale(outp,True)
        rinp = rescale(inp,False,0,1,0,255)
        errors.append(np.linalg.norm(inp-outp))
        #errors.append(HaarPSI(rinp,rout))
    plt.hist(errors)
    plt.show()
    #out = assemble_patches(outpatches,img2.shape)
    #hpi = HaarPSI(img2,out)
    #print('HaarPSI = %f ' % hpi)
    #psnrval = psnr(img2,out)
    #print('PSNR = %f  ' % psnrval)
    #twonorm = np.linalg.norm(img2-out,ord=2)
    #print('2 norm = %f' % twonorm)
    #fronorm = np.linalg.norm(img2-out,ord='fro')
    #print('Frobenius norm = %f' % fronorm)
    return(ocd,patches2,outpatches,errors)

class Patch():
    """Class representing a patch, i.e. small portion of an image"""
    
    def __init__(self,matrix):
        self.matrix = matrix.astype('float64')
        self.feature_matrix = None
        self.feature_vector = None

    def compute_feature_matrix(self,U,V):
        """Given matrices U and V computed by 2DPCA, computes the feature matrix of the patch"""

        self.feature_matrix = np.dot(np.dot(V.transpose(),self.matrix),U)

    def compute_feature_vector(self,eigenvectors):
        """Given matrix eigenvectors computed by PCA, vectorizes the patch and computes its feature vector"""
        
        length = self.matrix.flatten().shape[0]
        self.feature_vector = np.dot(self.matrix.flatten().reshape(1,length),eigenvectors)
        
    def __add__(self,patch):
        newmat = self.matrix + patch.matrix
        p = Patch(newmat)
        if self.feature_matrix is not None and patch.feature_matrix is not None:
            p.feature_matrix = self.feature_matrix + patch.feature_matrix
        if self.feature_vector is not None and patch.feature_vector is not None:
            p.feature_vector = self.feature_vector + patch.feature_vector
        return(p)

    def __mul__(self,scalar):
        newmat = scalar*self.matrix
        p = Patch(newmat)
        if self.feature_matrix is not None:
            p.feature_matrix = scalar*self.feature_matrix
        if self.feature_vector is not None:
            p.feature_vector = scalar*self.feature_vector
        return(p)
        
    def __rmul__(self,scalar):
        return(self.__mul__(scalar))
    
    def __truediv__(self,scalar):
        return(self.__mul__(1/scalar))

    def __sub__(self,other):
        return(self+((-1)*other))
    
    def show(self):
        fig = plt.figure()
        axis = fig.gca()
        plt.imshow(self.matrix, cmap=plt.cm.gray,interpolation='none')
        fig.show()

#class pca():
#    def __init__(self,patches=None,k=-1):
#        self.k = k
#        if patches is not None:
#            self.patches = tuple(patches)
#
#    def save_pickle(self,filepath):
#        f = open(filepath,'wb')
#        pickle.dump(self.__dict__,f,3)
#        f.close()
#
#    def load_pickle(self,filepath):
#        f = open(filepath,'rb')
#        tmpdict = pickle.load(f)
#        f.close()
#        self.__dict__.update(tmpdict)
#
#    def compute_pca(self):
#        length = self.patches[0].matrix.flatten().shape[0]
#        cov = covariance_matrix([x.matrix.flatten().reshape(length,1).transpose() for x in self.patches])
#        self.eigenvalues,self.eigenvectors = sslinalg.eigsh(cov,self.k)
#        
#

class twodpca():
    """Class representing an instance of the 2DPCA method"""
    
    def __init__(self,patches=None,l=-1,r=-1):
        self.l = l
        self.r = r
        self.bilateral_computed = False
        if patches is not None:
            self.patches = tuple(patches) #original image patches
                    
    def save_pickle(self,filepath):
        f = open(filepath,'wb')
        pickle.dump(self.__dict__,f,3)
        f.close()

    def load_pickle(self,filepath):
        f = open(filepath,'rb')
        tmpdict = pickle.load(f)
        f.close()
        self.__dict__.update(tmpdict)

    def compute_horizzontal_2dpca(self):
        cov = covariance_matrix([p.matrix for p in self.patches])
        eigenvalues,U = sslinalg.eigs(cov,self.l)
        self.horizzontal_eigenvalues = eigenvalues
        self.U = U
        
    def compute_vertical_2dpca(self):
        cov = covariance_matrix([p.matrix.transpose() for p in self.patches])
        eigenvalues,V = sslinalg.eigs(cov,self.r)
        self.vertical_eigenvalues = eigenvalues
        self.V = V
        
    def compute_simple_bilateral_2dpca(self):
        if self.bilateral_computed:
            raise Exception('Bilateral 2dpca already computed')
        self.compute_horizzontal_2dpca()
        self.compute_vertical_2dpca()
        self.bilateral_computed = True
    
    def compute_iterative_bilateral_2dpca(self):
        if self.bilateral_computed:
            raise Exception('Bilateral 2dpca already computed')
        #TODO: implement

        

class ocdict():
    def __init__(self, patches=None, filepath=None):
        if filepath is not None:
            self.load_pickle(filepath)
            return(None)
        if patches is not None:
            self.patches = tuple(patches) #original image patches
            self.npatches = len(patches)
            self.shape = self.patches[0].matrix.shape
            self.height,self.width = self.shape
        self.matrix_is_normalized = False
        self.feature_matrix_is_normalized = False
        self.root_node = None
        self.clustered = False
        self.matrix_computed = False
        self.feature_matrix_computed = False
        
    def save_pickle(self,filepath):
        f = open(filepath,'wb')
        pickle.dump(self.__dict__,f,3)
        f.close()

    def load_pickle(self,filepath):
        f = open(filepath,'rb')
        tmpdict = pickle.load(f)
        f.close()
        self.__dict__.update(tmpdict)

    def _compute_matrix(self):
        self.matrix = np.vstack([x.matrix.flatten() for x in self.dictelements]).transpose()
        self.matrix_computed = True
        self._normalize_matrix()


    def _compute_feature_matrix(self):
        self.feature_matrix = np.vstack([x.feature_matrix.flatten() for x in self.dictelements]).transpose()
        self.feature_matrix_computed = True
        
    def compute_cluster_centroids(self):
        self.cluster_centroids  = [(1/c.npatches)*sum([self.patches[k].matrix for k in c.patches_idx]) for c in self.leafs]

    def _distance_from_cluster(self,random_cluster=False):
        #if not hasattr(self,'cluster_centroids'):
        #    self.compute_cluster_centroids()
        if random_cluster:
            centroid1 = self.cluster_centroids[np.random.randint(len(self.cluster_centroids))]
            centroids = self.cluster_centroids.copy()
            centroids.remove(centroid1)
        else:
            centroid1 = self.cluster_centroids[0]
            centroids = self.cluster_centroids[1:]
        distances = -np.ones(len(centroids))
        for i,c in enumerate(centroids):
            distances[i] = np.linalg.norm(centroid1 - c)
        return(distances)
    
    def _normalize_matrix(self,ord=None):
        self.normalized_matrix = self.matrix - self.matrix.mean(axis=0)
        ncols = self.matrix.shape[1]
        self.normalization_coefficients = np.ones(shape=(self.matrix.shape[1],))
        for j in range(ncols):
            col = self.normalized_matrix[:,j]
            norm = np.linalg.norm(col,ord=ord)
            if norm != 0:
                col /= norm
                self.normalization_coefficients[j] = norm
        self.matrix_is_normalized = True

    def normalize_feature_matrix(self):
        #self.normalized_feature_matrix = np.copy(self.feature_matrix)
        self.normalized_feature_matrix = self.feature_matrix - self.feature_matrix.mean(axis=0) 
        ncols = self.feature_matrix.shape[1]
        self.fnormalization_coefficients = np.ones(shape=(self.feature_matrix.shape[1],))
        for j in range(ncols):
            col = self.normalized_feature_matrix[:,j]
            norm = np.linalg.norm(col)
            col /= norm
            self.fnormalization_coefficients[j] = norm
        self.feature_matrix_is_normalized = True
        
    def mutual_coherence(self):
        mc = 0
        mj1,mj2 = None,None
        ncols = self.matrix.shape[1]
        for j1 in range(ncols):
            for j2 in range(j1+1,ncols):
                col1 = self.matrix[:,j1]
                col2 = self.matrix[:,j2]
                sp = np.dot(col1,col2)
                sp /= np.linalg.norm(col1)*np.linalg.norm(col2)
                if sp > mc:
                    mj1 = j1
                    mj2 = j2
                    mc = sp
        return(mc,mj1,mj2)
        
    def monkey_cluster(self,levels=3):
        """Randomly clusters the data and creates Haar dictionary"""
        
        self.tree_depth = levels
        self.root_node = Node(tuple(np.arange(self.npatches)),None)
        tovisit = [self.root_node]
        next_tovisit = []
        dictelements = [sum([p.matrix for p in self.patches])]
        for l in range(levels):
            while len(tovisit) > 0 :
                cur = tovisit.pop()
                indexes = list(cur.patches_idx)
                np.random.shuffle(indexes)
                k = int(len(indexes)/2)
                lindexes = indexes[:k]
                rindexes = indexes[k:]
                child1 = Node(lindexes,cur)
                child2 = Node(rindexes,cur)
                next_tovisit.append(child1)
                next_tovisit.append(child2)
                centroid1 = sum([self.patches[i].matrix for i in lindexes])
                centroid2 = sum([self.patches[i].matrix for i in rindexes])
                curdict = centroid1/child1.npatches - centroid2/child2.npatches
                dictelements += [curdict]
            self.leafs = next_tovisit
            tovisit = next_tovisit
            next_tovisit = []
            
        self.dictelements = dictelements
        self.ndictelements = len(dictelements)
        self.clustered = True
        self.compute_cluster_centroids()
        return(self.dictelements)
            
    def twomeans_cluster(self,epsilon,minpatches=5,clusteronpatches=False):
        """Clusters data using recursive 2-means and creates Haar-dictionary"""
        
        self.root_node = Node(tuple(np.arange(self.npatches)),None)
        tovisit = []
        tovisit.append(self.root_node)
        dictelements = [(1/self.npatches)*sum(self.patches[1:],self.patches[0])] #global average
        self.leafs = []
        cur_nodes = set()
        prev_nodes = set()
        prev_nodes.add(self)
        depth = 0
        if clusteronpatches:
            self.cluster_data_matrix = np.vstack([p.matrix.flatten() for p in self.patches])
        else:
            self.cluster_data_matrix = np.vstack([p.feature_matrix.flatten() for p in self.patches])
            #self.cluster_data_matrix = np.vstack([p.feature_matrix.flatten()/np.linalg.norm(p.feature_matrix.flatten()) for p in self.patches])
        #deb = True
        while len(tovisit) > 0:
            cur = tovisit.pop()
            lpatches_idx = []
            rpatches_idx = []
            if cur.npatches > minpatches:
                km = KMeans(n_clusters=2).fit(self.cluster_data_matrix[np.array(cur.patches_idx)]) #TODO: add possibility to choose norm...
                if km.inertia_ > epsilon: #if km.inertia is still big, we branch on this node
                    for k,label in enumerate(km.labels_):
                        if label == 0:
                            lpatches_idx.append(cur.patches_idx[k])
                        if label == 1:
                            rpatches_idx.append(cur.patches_idx[k])
                    #if deb and (len(lpatches_idx) == 1 or len(rpatches_idx) == 1):
                    #    ipdb.set_trace()
                    #    deb = False
                    #centroid1 = (1/len(lpatches_idx))*sum([self.patches[i] for i in lpatches_idx[1:]],self.patches[lpatches_idx[0]])
                    #centroid2 = (1/len(rpatches_idx))*sum([self.patches[i] for i in rpatches_idx[1:]],self.patches[rpatches_idx[0]])
                    centroid1 = sum([self.patches[i] for i in lpatches_idx[1:]],self.patches[lpatches_idx[0]])
                    centroid2 = sum([self.patches[i] for i in rpatches_idx[1:]],self.patches[rpatches_idx[0]])
                    #centroid1 /= np.linalg.norm(centroid1.matrix)
                    #centroid2 /= np.linalg.norm(centroid2.lmatrix)
                    centroid1 /= len(lpatches_idx)
                    centroid2 /= len(rpatches_idx)
                    lnode = Node(lpatches_idx,cur,centroid1,True)
                    rnode = Node(rpatches_idx,cur,centroid2,False)
                    cur.children = (lnode,rnode)
                    #tovisit.append(lnode)
                    #tovisit.append(rnode)
                    tovisit = [lnode] + tovisit
                    tovisit = [rnode] + tovisit
                    depth = max((depth,lnode.depth,rnode.depth))
                    curdict = centroid1 - centroid2
                    curdict.matrix /= np.linalg.norm(curdict.matrix)
                    if curdict.feature_matrix is not None:
                        curdict.feature_matrix /= np.linalg.norm(curdict.feature_matrix)
                    #norm = np.linalg.norm(curdict.feature_matrix)
                    #if norm < 1.e-3:
                    #    ipdb.set_trace()
                    dictelements += [curdict]
            if cur.children is None:
                self.leafs.append(cur)
        self.dictelements = dictelements
        self.ndictelements = len(dictelements)
        self.tree_depth = depth
        self.clustered = True
        self.compute_cluster_centroids()
        self.tree_sparsity = len(self.leafs)/2**self.tree_depth
        return(self.dictelements)

    def twomeans_haar_cluster(self,epsilon,levels=3,minpatches=5):
        """Clusters data using recursive 2-means on Haar-transformed patches and creates Haar-dictionary"""

        #lev = int(np.log2(self.patches[0].matrix.shape[0]))
        for p in self.patches:
            p.haar_arr = pywt2array(pywt.wavedec2(p.matrix,'haar','periodic',levels))
        self.root_node = Node(tuple(np.arange(self.npatches)),None)
        tovisit = []
        tovisit.append(self.root_node)
        dictelements = [(1/self.npatches)*sum(self.patches[1:],self.patches[0])] #global average
        self.leafs = []
        cur_nodes = set()
        prev_nodes = set()
        prev_nodes.add(self)
        depth = 0
        self.cluster_data_matrix = np.vstack([p.haar_arr for p in self.patches])
        #deb = True
        while len(tovisit) > 0:
            cur = tovisit.pop()
            lpatches_idx = []
            rpatches_idx = []
            if cur.npatches > minpatches:
                km = KMeans(n_clusters=2).fit(self.cluster_data_matrix[np.array(cur.patches_idx)]) #TODO: add possibility to choose norm...
                if km.inertia_ > epsilon: #if km.inertia is still big, we branch on this node
                    for k,label in enumerate(km.labels_):
                        if label == 0:
                            lpatches_idx.append(cur.patches_idx[k])
                        if label == 1:
                            rpatches_idx.append(cur.patches_idx[k])
                    #if deb and (len(lpatches_idx) == 1 or len(rpatches_idx) == 1):
                    #    ipdb.set_trace()
                    #    deb = False
                    #centroid1 = (1/len(lpatches_idx))*sum([self.patches[i] for i in lpatches_idx[1:]],self.patches[lpatches_idx[0]])
                    #centroid2 = (1/len(rpatches_idx))*sum([self.patches[i] for i in rpatches_idx[1:]],self.patches[rpatches_idx[0]])
                    centroid1 = sum([self.patches[i] for i in lpatches_idx[1:]],self.patches[lpatches_idx[0]])
                    centroid2 = sum([self.patches[i] for i in rpatches_idx[1:]],self.patches[rpatches_idx[0]])
                    #centroid1 /= np.linalg.norm(centroid1.matrix)
                    #centroid2 /= np.linalg.norm(centroid2.lmatrix)
                    centroid1 /= len(lpatches_idx)
                    centroid2 /= len(rpatches_idx)
                    lnode = Node(lpatches_idx,cur,centroid1,True)
                    rnode = Node(rpatches_idx,cur,centroid2,False)
                    cur.children = (lnode,rnode)
                    #tovisit.append(lnode)
                    #tovisit.append(rnode)
                    tovisit = [lnode] + tovisit
                    tovisit = [rnode] + tovisit
                    depth = max((depth,lnode.depth,rnode.depth))
                    curdict = centroid1 - centroid2
                    curdict.matrix /= np.linalg.norm(curdict.matrix)
                    if curdict.feature_matrix is not None:
                        curdict.feature_matrix /= np.linalg.norm(curdict.feature_matrix)
                    #norm = np.linalg.norm(curdict.feature_matrix)
                    #if norm < 1.e-3:
                    #    ipdb.set_trace()
                    dictelements += [curdict]
            if cur.children is None:
                self.leafs.append(cur)
        self.dictelements = dictelements
        self.ndictelements = len(dictelements)
        self.tree_depth = depth
        self.clustered = True
        self.compute_cluster_centroids()
        self.tree_sparsity = len(self.leafs)/2**self.tree_depth
        return(self.dictelements)
    
    def spectral_cluster(self,epsilon,clusteronpatches=False,minpatches=5):
        """Clusters data using recursive spectral clustering and creates Haar-dictionary"""

        
        self.root_node = Node(tuple(np.arange(self.npatches)),None)
        tovisit = []
        tovisit.append(self.root_node)
        dictelements = [(1/self.npatches)*sum(self.patches[1:],self.patches[0])] #global average
        self.leafs = []
        cur_nodes = set()
        prev_nodes = set()
        prev_nodes.add(self)
        depth = 0
        nneighs = int(self.root_node.npatches/3)
        #affinity_matrix = kneighbors_graph(self.cluster_data_matrix, n_neighbors=nneighs, include_self=False,mode='distance')#.todense()
        #affinity_matrix.data = np.exp(-beta*affinity_matrix.data/std) + eps
        beta = 10
        eps = 0
        print('Computing affinity matrix...')
        #expaff_mat = np.zeros(shape=(self.npatches,self.npatches))
        expaff_mat = csr_matrix((self.npatches,self.npatches),dtype=np.float)
        thresh = 0.6
        if clusteronpatches:
            data = [p.matrix for p in self.patches]
        else:
            data = [p.feature_matrix for p in self.patches]
        for i in range(self.npatches):
            if i % 1000 == 0:
                print(i,len(expaff_mat.data))
            for j in range(i):
                #dist = np.linalg.norm(data[i] - data[j])
                #expdist = np.exp(-beta*dist) + eps
                expdist = HaarPSI(data[i],data[j])
                if expdist > thresh:
                    #print('yep')
                    expaff_mat[i,j] = expdist
                #expaff_mat[j,i] = expdist
        print('...done')
        expaff_mat = expaff_mat + expaff_mat.transpose()
        self.egvecs = []
                
        def Ncut(D,W,y):
            num = float(y.T.dot(D-W).dot(y))
            den = float(y.T.dot(D).dot(y))
            return(num/den)

        while len(tovisit) > 0:
            cur = tovisit.pop()
            lpatches_idx = []
            rpatches_idx = []
            aff_mat = expaff_mat[cur.patches_idx,:][:,cur.patches_idx]
            diag = np.array([row.sum() for row in aff_mat[:,:]])
            diagsqrt = np.diag(diag**(-1/2))
            mat = diagsqrt.dot(np.diag(diag) - aff_mat).dot(diagsqrt).astype('f')
            #print(depth, cur.npatches,aff_mat.shape)
            print('Computing eigenvalues/vectors of %s matrix' % mat.shape)
            egval,egvec = sslinalg.eigsh(mat,k=2,which='SM')
            #print("eigenvalues: ", egval)
            vec = egvec[:,1] #second eigenvalue
            #return(aff_mat,diag,mat,vec)
            leftrep = Patch(np.zeros_like(self.patches[0].matrix))
            #leftrep.feature_matrix = np.zeros_like(self.patches[0].feature_matrix)
            rightrep = Patch(np.zeros_like(self.patches[0].matrix))
            #rightrep.feature_matrix = np.zeros_like(self.patches[0].feature_matrix)
            #simple mean thresholding:
            #mean = vec.mean()
            #ipdb.set_trace()
            #isinleftcluster = vec > mean
            isinleftcluster = vec > filters.threshold_otsu(vec)
            #ipdb.set_trace()
            #ncutval = None
            #thresh_vals = vec.copy()
            #thresh_vals.sort()
            #for thresh_val in np.arange(vec.min(),vec.max(),(vec.max()-vec.min())/10):
            #step = int(len(vec)/10)
            #print("len vec = ", len(vec), "step = ", step)
            #for thresh_val in thresh_vals[-2::-step]:
            #for thresh_val in thresh_vals[::step]:
            #    cand_y = vec > thresh_val
            #    if cand_y.var() == 0: #cheap way to see if cand_y is constant
            #        continue
            #    cand_ncutval = Ncut(np.diag(diag),aff_mat,cand_y)
            #    print("Ncut = ", cand_ncutval,  "greater : ", len(cand_y.nonzero()[0]))
            #    if ncutval is None or cand_ncutval < ncutval:
            #        ncutval = cand_ncutval
            #        isinleftcluster = cand_y
            #        #y = cand_y
            touchA,touchB = (0,0)
            for k,label in np.ndenumerate(isinleftcluster):
                k = k[0]
                if label:
                    lpatches_idx.append(cur.patches_idx[k])
                    #leftrep += vec[k]*self.patches[cur.patches_idx[k]]
                    leftrep += self.patches[cur.patches_idx[k]]
                    touchA += 1
                else:
                    rpatches_idx.append(cur.patches_idx[k])
                    #rightrep += vec[k]*self.patches[cur.patches_idx[k]]
                    rightrep += self.patches[cur.patches_idx[k]]
                    touchB += 1
                #if (touchA >= 10 and touchB >= 10) and (leftrep.matrix.std() < 1.e-7 or  rightrep.matrix.std() < 1.e-7):
                #    ipdb.set_trace()
            leftrep.matrix /= np.linalg.norm(leftrep.matrix)
            rightrep.matrix /= np.linalg.norm(rightrep.matrix)
            print("left and right cards: ", touchA,touchB)
            #leftrep.feature_matrix /= np.linalg.norm(leftrep.feature_matrix)
            #rightrep.feature_matrix /= np.linalg.norm(rightrep.feature_matrix)
            leftvar = np.trace(covariance_matrix([self.patches[i].matrix.transpose() for i in lpatches_idx]))
            rightvar = np.trace(covariance_matrix([self.patches[i].matrix.transpose() for i in rpatches_idx]))
            minvar = min(leftvar,rightvar)
            print("left and rightvar: ", leftvar,rightvar)

            leftcentroid = (1/len(lpatches_idx))*sum([self.patches[i] for i in lpatches_idx[1:]],self.patches[lpatches_idx[0]])
            rightcentroid = (1/len(rpatches_idx))*sum([self.patches[i] for i in rpatches_idx[1:]],self.patches[rpatches_idx[0]])
            #leftspread = sum([np.linalg.norm(self.patches[i].feature_matrix-leftcentroid.feature_matrix) for i in lpatches_idx])
            #rightspread = sum([np.linalg.norm(self.patches[i].feature_matrix-rightcentroid.feature_matrix) for i in rpatches_idx])
            #leftspread = sum([np.linalg.norm(self.patches[i].matrix-leftcentroid.matrix) for i in lpatches_idx])
            #rightspread = sum([np.linalg.norm(self.patches[i].matrix-rightcentroid.matrix) for i in rpatches_idx])
            #minspread = min(leftspread,rightspread)
            #print('idstr = %10s leftspread = %4f rightspread = %4f leftvar = %4f rightvar = %4f' %(cur.idstr,leftspread,rightspread,leftvar,rightvar))
            #leftrep = leftcentroid
            #rightrep = rightcentroid
            #print(np.linalg.norm(leftcentroid.matrix - leftrep.matrix),np.linalg.norm(rightcentroid.matrix - rightrep.matrix))
            #if len(lpatches_idx) > 0 and len(rpatches_idx)> 0 and minvar > 1e-7:
            #if len(lpatches_idx) > 0 and len(rpatches_idx)> 0 and minspread > 1e-7:
            ncutval = Ncut(np.diag(diag),aff_mat,isinleftcluster)
            print("Ncut = ", ncutval)
            #ipdb.set_trace()
            if ncutval > epsilon:
                lnode = Node(lpatches_idx,cur,leftrep,True)
                rnode = Node(rpatches_idx,cur,rightrep,False)
                cur.children = (lnode,rnode)
                depth = max((depth,lnode.depth,rnode.depth))
                self.egvecs.append((depth,vec,isinleftcluster))
                curdict = leftrep - rightrep
                #norm = np.linalg.norm(curdict.feature_matrix)
                #if norm < 1.e-3:
                #    ipdb.set_trace()
                dictelements += [curdict]
                if len(lpatches_idx) > minpatches:
                    tovisit = [lnode] + tovisit
                if len(rpatches_idx) > minpatches:
                    tovisit = [rnode] + tovisit
            else:
                print('closing current branch')
            if cur.children is None:
                self.leafs.append(cur)
        self.dictelements = dictelements
        self.ndictelements = len(dictelements)
        self.tree_depth = depth
        self.clustered = True
        self.compute_cluster_centroids()
        self.tree_sparsity = len(self.leafs)/2**self.tree_depth
        return(self.dictelements)

    
    #def ward_cluster(self,minpatches=20):
    #    clusteronpatches = False
    #
    #    dictelements = [(1/self.npatches)*sum(self.patches[1:],self.patches[0])] #global average
    #    self.leafs = []
    #
    #    self.cluster_data_matrix = np.vstack([p.feature_matrix.flatten() for p in self.patches])
    #    ward = ward_tree(self.cluster_data_matrix)
    #    children = ward[0]
    #    self.root_node = Node(None,None)
    #    nodes = [None]*(children.max() + 2)
    #    nodes[-1] = self.root_node
    #
    #    #first pass to build tree 
    #    #for parentidx,childsidx in zip(range(len(children)*2-2,len(children)-1,-1),children[::-1]):
    #    for idx,childsidx in zip(range(len(children),0,-1),children[::-1]):
    #        parentidx = idx*2
    #        parent = nodes[parentidx]
    #        if parent is None:
    #            parent = Node(None,None)
    #            nodes[parentidx] = parent
    #        lchildidx,rchildidx = childsidx
    #        #if nodes[lchildidx] is not None or nodes[rchildidx] is not None:
    #        #    raise Exception("This shouldn't happen")
    #        if nodes[lchildidx] is not None:
    #            nodes[lchildidx].parent = parent
    #        else:
    #            lchild = Node(None,parent)                
    #        if nodes[rchildidx] is not None:
    #            nodes[rchildidx].parent = parent
    #            nodes[rchildidx].isleftchild = False
    #        else:
    #            rchild = Node(None,parent,isleftchild=False)
    #        if lchildidx <= len(children): #i.e. it's a leaf node
    #            lchild.patches_idx = (lchildidx,)
    #            lchild.npatches = 1
    #            self.leafs.append(lchild)
    #        if rchildidx <= len(children): #i.e. it's a leaf node
    #            rchild.patches_idx = (rchildidx,)
    #            rchild.npatches = 1
    #            self.leafs.append(rchild)
    #        parent.children = (lchild,rchild)
    #        nodes[lchildidx] = lchild
    #        nodes[rchildidx] = rchild
    #
    #    #second pass bottom-up to set patches_idx
    #    print('2')
    #    tovisit = set(self.leafs)
    #    while len(tovisit) > 0:
    #        cur = tovisit.pop()
    #        parent = cur.parent
    #        if parent.patches_idx is None:
    #            parent.patches_idx = cur.patches_idx
    #        else:
    #            parent.patches_idx += cur.patches_idx
    #        parent.npatches = len(parent.patches_idx)
    #        if parent.parent is not None: #if it's not the root node
    #            tovisit.add(parent)
    #
    #    #final pass to construct the dictionary
    #    print('3')
    #    tovisit = [self.root_node]
    #    depth = 0
    #    while len(tovisit) > 0:
    #        cur = tovisit.pop()
    #        if cur.npatches > minpatches:
    #            lchild,rchild = cur.children
    #            lchild.depth = cur.depth + 1
    #            rchild.depth = cur.depth + 1
    #            lchild.idstr += '0'
    #            rchild.idstr += '1'
    #            lpatches_idx,rpatches_idx = lchild.patches_idx,rchild.patches_idx
    #            centroid1 = (1/len(lpatches_idx))*sum([self.patches[i] for i in lpatches_idx[1:]],self.patches[lpatches_idx[0]])
    #            centroid2 = (1/len(rpatches_idx))*sum([self.patches[i] for i in rpatches_idx[1:]],self.patches[rpatches_idx[0]])
    #            tovisit = [lchild,rchild]
    #            depth = max((depth,lnode.depth,rnode.depth))
    #            curdict = centroid1 - centroid2
    #            dictelements += [curdict]
    #    
    #    self.dictelements = dictelements
    #    self.tree_depth = depth
    #    self.clustered = True
    #    self.compute_cluster_centroids()
    #    self.tree_sparsity = len(self.leafs)/2**self.tree_depth
    #    return(self.dictelements)

    
    def ksvd_dict(self,K,L,maxiter=5):
        """Computes dictionary using KSVD method. Requires KSVD.m file"""
        
        param = {'InitializationMethod': 'DataElements',
                 'K': K,
                 'L': L,
                 'displayProgress': 1,
                 'errorFlag': 0,
                 'numIteration': 10,
                 'preserveDCAtom': 1}
        length = self.patches[0].matrix.flatten().shape[0]
        Y = np.hstack([p.matrix.flatten().reshape(length,1) for p in self.patches])
        #octave.addpath('../ksvd')
        octave.addpath('ksvd')
        D = octave.KSVD(Y,param)
        self.matrix = D
        self.matrix_computed = True
        self.dictelements = []
        rows,cols = self.patches[0].matrix.shape
        for j in range(K):
            self.dictelements.append(Patch(D[:,j].reshape(rows,cols)))
        self.ndictelements = len(self.dictelements)
        
    def ksvdbox_dict(self,dictsize,sparsity,iternum=5):
        """Computes dictionary using KSVD method. Requires ksvd.m file from ksvdbox13"""

        octave.addpath('ksvdbox/')
        #octave.addpath('ompbox/') #TODO: BUG. from within ipython: first run uncomment, then comment. otherwise doesn't work....
        length = self.patches[0].matrix.flatten().shape[0]
        Y = np.hstack([p.matrix.flatten().reshape(length,1) for p in self.patches])
        params = {'data': Y,
                 'Tdata': sparsity,
                 'dictsize': dictsize,
                 'memusage': 'normal'} #'low','normal' or 'high'
        [D,X] = octave.ksvd(params)
        self.encoding = X
        self.matrix = D
        self.matrix_computed = True
        self.dictelements = []
        rows,cols = self.patches[0].matrix.shape
        for j in range(dictsize):
            self.dictelements.append(Patch(D[:,j].reshape(rows,cols)))
        self.ndictelements = len(self.dictelements)
        
    def sparse_code(self,input_patch,sparsity):
        """Uses OMP to sparsely code input_patch with the dictionary"""
        
        if input_patch.matrix.shape != self.patches[0].matrix.shape:
            raise Exception("Input patch is not of the correct size")
        if not self.matrix_computed:
            self._compute_matrix()
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
        y = input_patch.matrix.flatten()
        #mean = 0
        mean = np.mean(y)
        y -= mean
        #outnorm = np.linalg.norm(y)
        #matrix = self.matrix
        #normalize = self.matrix_is_normalized
        #if normalize:
        #    #y /= outnorm
        #    matrix = self.normalized_matrix
        #    norm_coefs = self.normalization_coefficients
        matrix = self.normalized_matrix
        omp.fit(matrix,y)
        coef = omp.coef_
        return(coef,mean)

    def sparse_code_ompext(self,input_patch,sparsity,ompbox=True):
        from oct2py import octave
        if ompbox:
            octave.addpath('ompbox')

        if input_patch.matrix.shape != self.patches[0].matrix.shape:
            raise Exception("Input patch is not of the correct size")
        if not self.matrix_computed:
            self._compute_matrix()
            
        y = input_patch.matrix.flatten()
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


    
    def decode(self,coefs,mean):
        #for idx in coefs.nonzero()[0]:
        #    out += coefs[idx]*self.normalized_matrix[:,idx].reshape(shape)
        out = (np.dot(self.normalized_matrix,coefs)).reshape(self.patches[0].matrix.shape)
        out += np.real(mean)
        return(out)

    def patches_subtree(self,startingnode):
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
    
    def show_dict(self,rows=10,cols=10):
        self.delm_by_var = sorted(self.dictelements,key=lambda x: np.var(x.matrix,axis=(0,1)),reverse=False)[:rows*cols]
        fig, axis = plt.subplots(rows,cols,sharex=True,sharey=True,squeeze=True)
        for idx, a in np.ndenumerate(axis):
            a.set_axis_off()
            a.imshow(self.delm_by_var[cols*idx[0] + idx[1]].matrix,interpolation='nearest')
        plt.show()

    #def show_dict_from_haar_matrix(self,rows=10,cols=10):
    #    levels = 3
    #    self.haar_recovered_dictelements = [pywt.waverec2(array2pywt(col,levels),'haar','periodic') for col in self.normalized_matrix.transpose()]
    #    self.delm_by_var = sorted(self.haar_recovered_dictelements,key=lambda x: np.var(x,axis=(0,1)),reverse=False)[:rows*cols]
    #    fig, axis = plt.subplots(rows,cols,sharex=True,sharey=True,squeeze=True)
    #    for idx, a in np.ndenumerate(axis):
    #        a.set_axis_off()
    #        a.imshow(self.delm_by_var[cols*idx[0] + idx[1]],interpolation='nearest')
    #    plt.show()

        
    def show_clusters(self,rows=10,cols=10,startingnode=None):
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
            centroid = sum([self.patches[i] for i in cur.patches_idx[1:]],self.patches[cur.patches_idx[0]])
            #centroid = sum([self.patches[i] for i in cur.patches_idx])
            if cur.idstr == '':
                string = 'root'
            else:
                string = cur.idstr
            clusters.append((centroid,string))
            if cur.children is not None:
                tovisit = [cur.children[0]] + tovisit
                tovisit = [cur.children[1]] + tovisit
        fig, axis = plt.subplots(rows,cols,sharex=True,sharey=True,squeeze=True)
        if len(axis.shape) == 2:
            for idx, a in np.ndenumerate(axis):
                a.set_axis_off()
                clust,string = clusters[rows*idx[0] + idx[1]]
                a.imshow(clust.matrix,interpolation='nearest')
                a.text(0,0,string,color='red')
        else:
            for idx, a in np.ndenumerate(axis):
                a.set_axis_off()
                clust,string = clusters[idx[0]]
                a.imshow(clust.matrix,interpolation='nearest')
                a.text(0,0,string,color='red')
        plt.show()
        return(clusters)
    
class Node():
    
    def __init__(self,patches_idx,parent,centroid=None,isleftchild=True):
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
        self.centroid = centroid
        self.children = None
        self.patches_idx = None
        if patches_idx is not None:
            self.patches_idx = tuple(patches_idx) #indexes of d.patches_idx where d is an ocdict
            self.npatches= len(self.patches_idx)

    def patches_list(self,ocdict):
        return([ocdict.patches[i] for i in self.patches_idx])

