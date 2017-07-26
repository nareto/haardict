import ipdb
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as sslinalg
import skimage.io
import scipy.sparse
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import ward_tree
from sklearn.neighbors import kneighbors_graph
from oct2py import octave
import gc
import queue

def rescale(img,newmin=0,newmax=255):
    curmin,curmax = img.min(),img.max()
    angcoeff = (newmax-newmin)/(curmax-curmin)
    f = lambda x: newmax+angcoeff*(x-curmax)
    out = np.zeros_like(img)
    for idx,val in np.ndenumerate(img):
        out[idx] = f(val)
    return(out)

def clip(img):
    out = img.copy()
    out[out < 0] = 0
    out[out > 255] = 255
    return(out)

def stack(mat):
    out = np.hstack(np.vsplit(mat,mat.shape[0])).transpose()
    return(out)

def psnr(img1,img2):
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
    haarpsi = octave.HaarPSI(img1,img2)
    return(haarpsi)


def affinity_matrix(X,sigma=1,threshold=0.5):
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
    if len(values) == 0:
        raise Exception("Can't compute centroid of void set")
    centroid = 0
    for val in values:
        centroid += val
    centroid /= len(values)
    return(centroid)

def twomeansval(values,k):
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
    prev_v = values[0]
    for v in values[1:]:
        if v < prev_v:
            return(False)
    return(True)

def oneDtwomeans(values):
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
    

#def compute_dict_from_tree(root):
#    avg = sum(root.patches)/root.npatches
#    dictionary = [avg] + root.compute_tree()

def extract_patches(array,size=(8,8)):
    ret = []
    height,width = array.shape
    vstep,hstep = size
    for j in range(0,width-hstep+1,hstep):
        for i in range(0,height-vstep+1,vstep):
            subimg = array[i:i+vstep,j:j+hstep]
            ret.append(subimg)
    return(ret)

def assemble_patches(patches,out_size):
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
    centroid = sum(patches)/len(patches)
    p = patches[0]
    ret = np.zeros_like(np.dot(p.transpose(),p))
    for patch in patches:
        addend = np.dot((patch - centroid).transpose(),(patch-centroid))
        ret += addend
    ret /= len(patches)
    return(ret)

def learn_dict(paths,l=3,r=3,cluster_epsilon=1.e-2):
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
    twodpca_instance = twodpca(patches,l,r)
    twodpca_instance.compute_simple_bilateral_2dpca()

    for p in patches:
        p.compute_feature_matrix(twodpca_instance.U,twodpca_instance.V)

    ocd = ocdict(patches)
    ocd.twomeans_cluster(epsilon=cluster_epsilon)

    return(twodpca_instance,ocd)

def learn_dict_pca(paths,k=2):
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
    pca_instance = pca(patches,k)
    pca_instance.compute_pca()
    for p in patches:
        p.compute_feature_vector(pca_instance.eigenvectors)
    ocd = ocdict(patches)
    ocd.twomeans_cluster(twomeans_on_patches,haar_dict_on_patches)

    return(twodpca_instance,ocd)

def learn_dict_ksvd(paths,ndictelements,T0):
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
    ocd = ocdict(patches)
    ocd.ksvd_dict(ndictelements,T0)

    return(ocd)

def learn_dict_spectral(paths,l=3,r=3):
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
    twodpca_instance = twodpca(patches,l,r)
    twodpca_instance.compute_simple_bilateral_2dpca()

    for p in patches:
        p.compute_feature_matrix(twodpca_instance.U,twodpca_instance.V)

    ocd = ocdict(patches)
    ocd.spectral_cluster2()

    return(twodpca_instance,ocd)

def learn_dict_ward_tree(paths,l=3,r=3):
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
    twodpca_instance = twodpca(patches,l,r)
    twodpca_instance.compute_simple_bilateral_2dpca()

    for p in patches:
        p.compute_feature_matrix(twodpca_instance.U,twodpca_instance.V)

    ocd = ocdict(patches)
    ocd.ward_cluster()

    return(twodpca_instance,ocd)

class Patch():
    def __init__(self,matrix):
        self.matrix = matrix.astype('float64')
        self.feature_matrix = None
        self.feature_vector = None

    def compute_feature_matrix(self,U,V):
        self.feature_matrix = np.dot(np.dot(V.transpose(),self.matrix),U)

    def compute_feature_vector(self,eigenvectors):
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

class pca():
    def __init__(self,patches=None,k=-1):
        self.k = k
        if patches is not None:
            self.patches = tuple(patches)

    def save_pickle(self,filepath):
        f = open(filepath,'wb')
        pickle.dump(self.__dict__,f,3)
        f.close()

    def load_pickle(self,filepath):
        f = open(filepath,'rb')
        tmpdict = pickle.load(f)
        f.close()
        self.__dict__.update(tmpdict)

    def compute_pca(self):
        length = self.patches[0].matrix.flatten().shape[0]
        cov = covariance_matrix([x.matrix.flatten().reshape(length,1).transpose() for x in self.patches])
        self.eigenvalues,self.eigenvectors = sslinalg.eigsh(cov,self.k)
        
            
class twodpca():
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
        
    def monkey_cluster(self,levels=3):
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
        self.clustered = True
        self.compute_cluster_centroids()
        return(self.dictelements)
            
    def slink_kmeans_cluster(self):
        pass

    def twomeans_cluster(self,minpatches=5,epsilon=1.e-2,pca=False):
        clusteronpatches = False
        self.root_node = Node(tuple(np.arange(self.npatches)),None)
        tovisit = []
        tovisit.append(self.root_node)
        dictelements = [(1/self.npatches)*sum(self.patches[1:],self.patches[0])] #global average
        self.leafs = []
        cur_nodes = set()
        prev_nodes = set()
        prev_nodes.add(self)
        depth = 0
        if pca:
            data_matrix = np.vstack([p.feature_vector for p in self.patches]) 
        elif clusteronpatches:
            data_matrix = np.vstack([p.matrix.flatten() for p in self.patches])
        else:
            data_matrix = np.vstack([p.feature_matrix.flatten() for p in self.patches]) 
        while len(tovisit) > 0:
            cur = tovisit.pop()
            lpatches_idx = []
            rpatches_idx = []
            if cur.npatches > minpatches:
                km = KMeans(n_clusters=2).fit(data_matrix[np.array(cur.patches_idx)])
                if km.inertia_ > epsilon: 
                    for k,label in enumerate(km.labels_):
                        if label == 0:
                            lpatches_idx.append(cur.patches_idx[k])
                        if label == 1:
                            rpatches_idx.append(cur.patches_idx[k])
                    #centroid1 = (1/len(lpatches_idx))*sum([self.patches[i] for i in lpatches_idx[1:]],self.patches[lpatches_idx[0]])
                    #centroid2 = (1/len(rpatches_idx))*sum([self.patches[i] for i in rpatches_idx[1:]],self.patches[rpatches_idx[0]])
                    centroid1 = sum([self.patches[i] for i in lpatches_idx[1:]],self.patches[lpatches_idx[0]])
                    centroid2 = sum([self.patches[i] for i in rpatches_idx[1:]],self.patches[rpatches_idx[0]])
                    centroid1 /= np.linalg.norm(centroid1.matrix)
                    centroid2 /= np.linalg.norm(centroid2.matrix)
                    lnode = Node(lpatches_idx,cur,centroid1,True)
                    rnode = Node(rpatches_idx,cur,centroid2,False)
                    cur.children = (lnode,rnode)
                    #tovisit.append(lnode)
                    #tovisit.append(rnode)
                    tovisit = [lnode] + tovisit
                    tovisit = [rnode] + tovisit
                    depth = max((depth,lnode.depth,rnode.depth))
                    curdict = centroid1 - centroid2
                    #norm = np.linalg.norm(curdict.feature_matrix)
                    #if norm < 1.e-3:
                    #    ipdb.set_trace()
                    dictelements += [curdict]
            if cur.children is None:
                self.leafs.append(cur)
        self.dictelements = dictelements
        self.tree_depth = depth
        self.clustered = True
        self.compute_cluster_centroids()
        self.tree_sparsity = len(self.leafs)/2**self.tree_depth
        return(self.dictelements)

    def spectral_cluster(self,minpatches=50):
        clusteronpatches = False
        self.root_node = Node(tuple(np.arange(self.npatches)),None)
        tovisit = []
        tovisit.append(self.root_node)
        dictelements = [(1/self.npatches)*sum(self.patches[1:],self.patches[0])] #global average
        self.leafs = []
        cur_nodes = set()
        prev_nodes = set()
        prev_nodes.add(self)
        data_matrix = np.vstack([p.feature_matrix.flatten() for p in self.patches])
        depth = 0
        nneighs = int(self.root_node.npatches/3)
        affinity_matrix = kneighbors_graph(data_matrix, n_neighbors=nneighs, include_self=False,mode='distance').todense()
        beta = 5
        eps = 0
        affinity_matrix = np.exp(-beta * affinity_matrix /affinity_matrix[affinity_matrix.nonzero()].std()) + eps
        while len(tovisit) > 0:
            cur = tovisit.pop()
            lpatches_idx = []
            rpatches_idx = []
            if cur.npatches > minpatches: #TODO: better stop criteria based on NCut value
                #cur_data = data_matrix[np.array(cur.patches_idx)]
                #cur_data = data_matrix[cur.patches_idx,:]
                aff_mat = affinity_matrix[cur.patches_idx,:][:,cur.patches_idx]
                aff_mat = 0.5*(aff_mat + aff_mat.transpose())
                print(depth, cur.npatches,aff_mat.shape)
                sc = SpectralClustering(n_clusters=2, eigen_solver='arpack',affinity="precomputed",assign_labels='discretize')
                sc.fit(aff_mat)
                for k,label in enumerate(sc.labels_):
                    if label == 0:
                        lpatches_idx.append(cur.patches_idx[k])
                    if label == 1:
                        rpatches_idx.append(cur.patches_idx[k])
                if len(lpatches_idx) > 0 and len(rpatches_idx)> 0:
                    centroid1 = (1/len(lpatches_idx))*sum([self.patches[i] for i in lpatches_idx[1:]],self.patches[lpatches_idx[0]])
                    centroid2 = (1/len(rpatches_idx))*sum([self.patches[i] for i in rpatches_idx[1:]],self.patches[rpatches_idx[0]])
                    lnode = Node(lpatches_idx,cur,centroid1,True)
                    rnode = Node(rpatches_idx,cur,centroid2,False)
                    cur.children = (lnode,rnode)
                    #tovisit.append(lnode)
                    #tovisit.append(rnode)
                    tovisit = [lnode] + tovisit
                    tovisit = [rnode] + tovisit
                    depth = max((depth,lnode.depth,rnode.depth))
                    curdict = centroid1 - centroid2
                    #norm = np.linalg.norm(curdict.feature_matrix)
                    #if norm < 1.e-3:
                    #    ipdb.set_trace()
                    dictelements += [curdict]
            if cur.children is None:
                self.leafs.append(cur)
        self.dictelements = dictelements
        self.tree_depth = depth
        self.clustered = True
        self.compute_cluster_centroids()
        self.tree_sparsity = len(self.leafs)/2**self.tree_depth
        return(self.dictelements)

    def spectral_cluster2(self,minpatches=50):
        clusteronpatches = False
        self.root_node = Node(tuple(np.arange(self.npatches)),None)
        tovisit = []
        tovisit.append(self.root_node)
        dictelements = [(1/self.npatches)*sum(self.patches[1:],self.patches[0])] #global average
        self.leafs = []
        cur_nodes = set()
        prev_nodes = set()
        prev_nodes.add(self)
        data_matrix = np.vstack([p.feature_matrix.flatten() for p in self.patches])
        depth = 0
        nneighs = int(self.root_node.npatches/10)
        affinity_matrix = kneighbors_graph(data_matrix, n_neighbors=nneighs, include_self=False,mode='distance')#.todense()
        beta = 5
        eps = 0
        dadj = affinity_matrix.todense()
        std = dadj[dadj.nonzero()].std()
        affinity_matrix.data = np.exp(-beta*affinity_matrix.data/std) + eps
        while len(tovisit) > 0:
            cur = tovisit.pop()
            lpatches_idx = []
            rpatches_idx = []
            if cur.npatches > minpatches: #TODO: better stop criteria based on NCut value
                #cur_data = data_matrix[np.array(cur.patches_idx)]
                #cur_data = data_matrix[cur.patches_idx,:]
                aff_mat = affinity_matrix[cur.patches_idx,:][:,cur.patches_idx]
                aff_mat = 0.5*(aff_mat + aff_mat.transpose())
                diag = np.array([row.sum() for row in aff_mat[:,:]])
                diagsqrt = np.diag(diag**(-1/2))
                mat = diagsqrt.dot(np.diag(diag) - aff_mat).dot(diagsqrt).astype('f')
                print(depth, cur.npatches,aff_mat.shape)
                egval,egvec = sslinalg.eigsh(mat,k=2,which='SM')
                vec = egvec[:,1] #second eigenvalue
                leftrep = Patch(np.zeros_like(self.patches[0].matrix))
                rightrep = Patch(np.zeros_like(self.patches[0].matrix))
                #simple mean thresholding:
                mean = vec.mean()
                isinleftcluster = vec > mean
                touchA,touchB = (0,0)
                for k,label in np.ndenumerate(isinleftcluster):
                    k = k[0]
                    if label:
                        lpatches_idx.append(cur.patches_idx[k])
                        #leftrep += vec[k]*self.patches[k]
                        leftrep += self.patches[k]
                        touchA += 1
                    else:
                        rpatches_idx.append(cur.patches_idx[k])
                        #rightrep += vec[k]*self.patches[k]
                        rightrep += self.patches[k]
                        touchB += 1
                    #if (touchA >= 10 and touchB >= 10) and (leftrep.matrix.std() < 1.e-7 or  rightrep.matrix.std() < 1.e-7):
                    #    ipdb.set_trace()
                leftrep /= np.linalg.norm(leftrep.matrix)
                rightrep /= np.linalg.norm(rightrep.matrix)
                if len(lpatches_idx) > 0 and len(rpatches_idx)> 0:
                    lnode = Node(lpatches_idx,cur,leftrep,True)
                    rnode = Node(rpatches_idx,cur,rightrep,False)
                    cur.children = (lnode,rnode)
                    #tovisit.append(lnode)
                    #tovisit.append(rnode)
                    tovisit = [lnode] + tovisit
                    tovisit = [rnode] + tovisit
                    depth = max((depth,lnode.depth,rnode.depth))
                    curdict = leftrep - rightrep
                    #norm = np.linalg.norm(curdict.feature_matrix)
                    #if norm < 1.e-3:
                    #    ipdb.set_trace()
                    dictelements += [curdict]
            if cur.children is None:
                self.leafs.append(cur)
        self.dictelements = dictelements
        self.tree_depth = depth
        self.clustered = True
        self.compute_cluster_centroids()
        self.tree_sparsity = len(self.leafs)/2**self.tree_depth
        return(self.dictelements)

    
    def ward_cluster(self,minpatches=20):
        clusteronpatches = False
    
        dictelements = [(1/self.npatches)*sum(self.patches[1:],self.patches[0])] #global average
        self.leafs = []
    
        data_matrix = np.vstack([p.feature_matrix.flatten() for p in self.patches])
        ward = ward_tree(data_matrix)
        children = ward[0]
        self.root_node = Node(None,None)
        nodes = [None]*(children.max() + 2)
        nodes[-1] = self.root_node

        #first pass to build tree 
        #for parentidx,childsidx in zip(range(len(children)*2-2,len(children)-1,-1),children[::-1]):
        for idx,childsidx in zip(range(len(children),0,-1),children[::-1]):
            parentidx = idx*2
            parent = nodes[parentidx]
            if parent is None:
                parent = Node(None,None)
                nodes[parentidx] = parent
            lchildidx,rchildidx = childsidx
            #if nodes[lchildidx] is not None or nodes[rchildidx] is not None:
            #    raise Exception("This shouldn't happen")
            if nodes[lchildidx] is not None:
                nodes[lchildidx].parent = parent
            else:
                lchild = Node(None,parent)                
            if nodes[rchildidx] is not None:
                nodes[rchildidx].parent = parent
                nodes[rchildidx].isleftchild = False
            else:
                rchild = Node(None,parent,isleftchild=False)
            if lchildidx <= len(children): #i.e. it's a leaf node
                lchild.patches_idx = (lchildidx,)
                lchild.npatches = 1
                self.leafs.append(lchild)
            if rchildidx <= len(children): #i.e. it's a leaf node
                rchild.patches_idx = (rchildidx,)
                rchild.npatches = 1
                self.leafs.append(rchild)
            parent.children = (lchild,rchild)
            nodes[lchildidx] = lchild
            nodes[rchildidx] = rchild

        #second pass bottom-up to set patches_idx
        print('2')
        tovisit = set(self.leafs)
        while len(tovisit) > 0:
            cur = tovisit.pop()
            parent = cur.parent
            if parent.patches_idx is None:
                parent.patches_idx = cur.patches_idx
            else:
                parent.patches_idx += cur.patches_idx
            parent.npatches = len(parent.patches_idx)
            if parent.parent is not None: #if it's not the root node
                tovisit.add(parent)

        #final pass to construct the dictionary
        print('3')
        tovisit = [self.root_node]
        depth = 0
        while len(tovisit) > 0:
            cur = tovisit.pop()
            if cur.npatches > minpatches:
                lchild,rchild = cur.children
                lchild.depth = cur.depth + 1
                rchild.depth = cur.depth + 1
                lchild.idstr += '0'
                rchild.idstr += '1'
                lpatches_idx,rpatches_idx = lchild.patches_idx,rchild.patches_idx
                centroid1 = (1/len(lpatches_idx))*sum([self.patches[i] for i in lpatches_idx[1:]],self.patches[lpatches_idx[0]])
                centroid2 = (1/len(rpatches_idx))*sum([self.patches[i] for i in rpatches_idx[1:]],self.patches[rpatches_idx[0]])
                tovisit = [lchild,rchild]
                depth = max((depth,lnode.depth,rnode.depth))
                curdict = centroid1 - centroid2
                dictelements += [curdict]
        
        self.dictelements = dictelements
        self.tree_depth = depth
        self.clustered = True
        self.compute_cluster_centroids()
        self.tree_sparsity = len(self.leafs)/2**self.tree_depth
        return(self.dictelements)

    
    def ksvd_dict(self,K,L,maxiter=5):
        param = {'InitializationMethod': 'DataElements',
                 'K': K,
                 'L': L,
                 'displayProgress': 1,
                 'errorFlag': 0,
                 'numIteration': 10,
                 'preserveDCAtom': 1}
        length = self.patches[0].matrix.flatten().shape[0]
        Y = np.hstack([p.matrix.flatten().reshape(length,1) for p in self.patches])
        octave.addpath('../ksvd')
        D = octave.KSVD(Y,param)
        self.matrix = D
        self.matrix_computed = True
        self.dictelements = []
        rows,cols = self.patches[0].matrix.shape
        for j in range(K):
            self.dictelements.append(Patch(D[:,j].reshape(rows,cols)))
    
    def sparse_code(self,input_patch,sparsity):
        if input_patch.matrix.shape != self.patches[0].matrix.shape:
            raise Exception("Input patch is not of the correct size")
        if not self.matrix_computed:
            self._compute_matrix()
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
        y = input_patch.matrix.flatten()
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

    def sparse_code_tree(self,input_patch,distonpatches,sparsity=-1):
        #data_matrix = np.vstack([p.feature_matrix.flatten() for p in self.patches])
        cur_node = self.root_node
        next_node = None
        point = input_patch
        coef = np.zeros(len(self.dictelements))
        input_patch.matrix -= input_patch.matrix.mean()
        input_patch.feature_matrix -= input_patch.feature_matrix.mean()
        while cur_node.children is not None:
            left_dist = np.linalg.norm(point.matrix - cur_node.children[0].centroid.matrix)
            right_dist = np.linalg.norm(point.matrix - cur_node.children[1].centroid.matrix)
            left_dist_feat = np.linalg.norm(point.feature_matrix - cur_node.children[0].centroid.feature_matrix)
            right_dist_feat = np.linalg.norm(point.feature_matrix - cur_node.children[1].centroid.feature_matrix)
            if distonpatches:
                ld = left_dist
                rd = right_dist
            else:
                ld = left_dist_feat
                rd = right_dist_feat
            if ld < rd:
                marker = 'L'
                next_node = cur_node.children[0]
                print(left_dist,left_dist_feat)
            else:
                marker = 'R'
                next_node = cur_node.children[1]
                print(right_dist,right_dist_feat)
            #print(marker+' left: %3f right %3f' % (left_dist,right_dist))
            cur_node = next_node
        #print(cur_node.depth)
        return(cur_node.centroid)

    #def haar_tree_decode(self):
    #    tovisit = [self.root_node]
    #    prev_centroid = self.dictelements[0]
    #    decoded_centroid = [prev_centroid]
    #    while tovisit: #WRONG CONDITION
    #        cur_node = tovisit.pop()
    #        tovisit = [cur_node.children[0],cur_node.children[1]]+ tovisit
    #        decoded_centroid += [prev_centroid ]
    #    #return(cur_node.centroid)
    
    
    def decode(self,coefs,mean):
        #for idx in coefs.nonzero()[0]:
        #    out += coefs[idx]*self.normalized_matrix[:,idx].reshape(shape)
        out = (np.dot(self.normalized_matrix,coefs)).reshape(self.patches[0].matrix.shape)
        out += np.real(mean)
        return(out)

    def show_dict(self,rows=10,cols=10):
        self.delm_by_var = sorted(self.dictelements,key=lambda x: np.var(x.matrix,axis=(0,1)),reverse=False)[:rows*cols]
        fig, axis = plt.subplots(rows,cols,sharex=True,sharey=True,squeeze=True)
        for idx, a in np.ndenumerate(axis):
            a.set_axis_off()
            a.imshow(self.delm_by_var[rows*idx[0] + idx[1]].matrix,interpolation='nearest')
        plt.show()

    def show_clusters(self,rows=10,cols=10,startingnode=None):
        clusters = []
        if startingnode is None:
            tovisit = [self.root_node]
        else:
            curn = self.root_node
            for c in startingnode:
                nextn = curn.children[int(c)]
            tovisit = [nextn]
        while len(tovisit) > 0:
            cur = tovisit.pop()
            centroid = sum([self.patches[i] for i in cur.patches_idx[1:]],self.patches[cur.patches_idx[0]])
            #centroid = sum([self.patches[i] for i in cur.patches_idx])
            clusters.append(centroid)
            if cur.children is not None:
                tovisit = [cur.children[0]] + tovisit
                tovisit = [cur.children[1]] + tovisit
        #self.clust_by_var = sorted(clusters,key=lambda x: np.var(x.matrix,axis=(0,1)),reverse=True)[:rows*cols]
        self.clust_by_var = clusters
        fig, axis = plt.subplots(rows,cols,sharex=True,sharey=True,squeeze=True)
        if len(axis.shape) == 2:
            for idx, a in np.ndenumerate(axis):
                a.set_axis_off()
                a.imshow(self.clust_by_var[rows*idx[0] + idx[1]].matrix,interpolation='nearest')
        else:
            for idx, a in np.ndenumerate(axis):
                a.set_axis_off()
                a.imshow(self.clust_by_var[idx[0]].matrix,interpolation='nearest')

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

