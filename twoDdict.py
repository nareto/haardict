import ipdb
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as sslinalg
import skimage.io
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.cluster import KMeans
from oct2py import octave
import gc
import queue

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
    return(20*np.log10(255/mse))

def HaarPSI(img1,img2):
    """Computes HaarPSI of img1 vs. img2. Requires file HaarPSI.m to be present in working directory"""
    from oct2py import octave
    img1 = img1.astype('float64')
    img2 = img2.astype('float64')
    octave.eval('pkg load image')
    haarpsi = octave.HaarPSI(img1,img2)
    return(haarpsi)

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

def learn_dict(paths,l=2,r=2):
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
    ocd.twomeans_cluster()

    return(twodpca_instance,ocd)


class Patch():
    def __init__(self,matrix):
        self.matrix = matrix.astype('float64')
        self.feature_matrix = None

    def compute_feature_matrix(self,U,V):
        self.feature_matrix = np.dot(np.dot(V.transpose(),self.matrix),U)
        
    def show(self):
        fig = plt.figure()
        axis = fig.gca()
        plt.imshow(self.matrix, cmap=plt.cm.gray,interpolation='none')
        fig.show()

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
    def __init__(self, patches=None):
        if patches is not None:
            self.patches = tuple(patches) #original image patches
            self.npatches = len(patches)
            self.shape = self.patches[0].matrix.shape
            self.height,self.width = self.shape
        #self.matrix = np.hstack([np.hstack(np.vsplit(x,self.height)).transpose() for x in self.patches])
        #self.normalization_coefficients = np.ones(shape=(self.matrix.shape[1],))
        self.matrix_is_normalized = False
        #self.normalize_matrix()
        #self.sparse_coefs = None
        self.root_node = None
        self.clustered = False
        self.matrix_computed = False
        
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
        if not self.clustered:
            raise Exception("You need to cluster the patches first")
        #self.matrix = np.hstack([np.hstack(np.vsplit(x,self.height)).transpose() for x in self.dictelements])
        self.matrix = np.vstack([x.flatten() for x in self.dictelements]).transpose()
        self.matrix_computed = True
        #self.normalization_coefficients = np.ones(shape=(self.matrix.shape[1],))
        #self.matrix_is_normalized = False
        #self.normalize_matrix()

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
    
    def normalize_matrix(self):
        self.normalized_matrix = np.copy(self.matrix)
        ncols = self.matrix.shape[1]
        self.normalization_coefficients = np.ones(shape=(self.matrix.shape[1],))
        for j in range(ncols):
            col = self.normalized_matrix[:,j]
            norm = np.linalg.norm(col)
            col /= norm
            self.normalization_coefficients[j] = norm
        self.matrix_is_normalized = True

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

    def twomeans_cluster(self,minpatches=5,epsilon=1.e-2):
        self.root_node = Node(tuple(np.arange(self.npatches)),None)
        #tovisit = queue.Queue()
        tovisit = []
        #tovisit.put(self.root_node)
        tovisit.append(self.root_node)
        dictelements = [sum([p.matrix for p in self.patches])] #global average
        self.leafs = []
        cur_nodes = set()
        prev_nodes = set()
        prev_nodes.add(self)
        depth = 0
        patches_matrix = np.vstack([p.matrix.flatten() for p in self.patches]) 
        #while not tovisit.empty():
        while len(tovisit) > 0:
            #cur = tovisit.get()
            cur = tovisit.pop()
            lpatches_idx = []
            rpatches_idx = []
            if cur.npatches > minpatches:
                km = KMeans(n_clusters=2).fit(patches_matrix[np.array(cur.patches_idx)])
                if km.inertia_ > epsilon: #TODO: check values
                    for k,label in enumerate(km.labels_):
                        if label == 0:
                            lpatches_idx.append(cur.patches_idx[k])
                        if label == 1:
                            rpatches_idx.append(cur.patches_idx[k])
                    lnode = Node(lpatches_idx,cur)
                    rnode = Node(rpatches_idx,cur)
                    cur.children = (lnode,rnode)
                    #tovisit.put(lnode)
                    #tovisit.put(rnode)
                    tovisit.append(lnode)
                    tovisit.append(rnode)
                    depth = max((depth,lnode.depth,rnode.depth))
                    centroid1 = sum([self.patches[i].matrix for i in lpatches_idx])
                    centroid2 = sum([self.patches[i].matrix for i in rpatches_idx])
                    curdict = centroid1/lnode.npatches - centroid2/rnode.npatches
                    if np.linalg.norm(curdict) < 1.e-3:
                        ipdb.set_trace()
                    dictelements += [curdict]
                    
            if len(lpatches_idx) == 0 and len(rpatches_idx) == 0: #?better if cur.children is None
                self.leafs.append(cur)
        self.dictelements = dictelements
        self.tree_depth = depth
        self.clustered = True
        self.compute_cluster_centroids()
        return(self.dictelements)

    def sparse_code(self,input_patch,sparsity):
        if input_patch.matrix.shape != self.patches[0].matrix.shape:
            raise Exception("Input patch is not of the correct size")
        if not self.matrix_computed:
            self._compute_matrix()
            self.normalize_matrix()
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
        #omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity,normalize=True)
        #omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity,fit_intercept=True,normalize=True,tol=1)
        #omp = OrthogonalMatchingPursuit(fit_intercept=True,normalize=True,tol=1)
        #y = np.hstack(np.vsplit(input_patch.matrix,self.height)).transpose()
        y = input_patch.matrix.flatten()
        mean = np.mean(y)
        y -= mean
        if self.matrix_is_normalized:
            #omp.fit(self.normalized_matrix,y)
            #coef = omp.coef_
            yoct = y.astype('float64').transpose()
            dictoct = self.normalized_matrix.astype('float64')
            #gramdict = dictoct.transpose().dot(dictoct)
            coef = octave.OMP(sparsity,yoct,dictoct).transpose()
            #coef = octave.ompdChol(yoct,dictoct,gramdict,dictoct.transpose().dot(yoct),sparsity,1.e-3).transpose()
            for idx,norm in np.ndenumerate(self.normalization_coefficients):
                coef[idx] /= norm
        else:
            omp.fit(self.matrix,y)
            coef = omp.coef_
            #coef = octave.OMP(sparsity,y.astype('float64').transpose(),self.matrix.astype('float64')).transpose()
        #coef += mean
        #print('Error in sparse coding: %f' % np.linalg.norm(input_patch.matrix - self.decode(coef)))
        return(coef,mean)

    def decode(self,coefs,mean):
        out = np.zeros_like(self.patches[0].matrix)
        for idx in coefs.nonzero()[0]:
            #out += coefs[idx]*self.patches[idx]
            out += coefs[idx]*self.matrix[:,idx].reshape(self.shape)
            #out += coefs[idx]*self.normalized_matrix[:,idx].reshape(self.shape)
        #if out.min() < 0:
        #    out -= out.min()
        #out /= out.max()
        return(out+mean)
    
class Node():
    def __init__(self,patches_idx,parent):
        self.parent = parent
        if parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1
        self.children = None
        self.patches_idx = tuple(patches_idx) #indexes of d.patches_idx where d is an ocdict
        self.npatches= len(self.patches_idx)

    def patches_list(self,ocdict):
        return([ocdict.patches[i] for i in self.patches_idx])

    #def build_patches_card_list(self):
    #    out = []
    #    for leaf in self.leafs:
    #        out.append(leaf.npatches)
    #    return(out)
    #    
    #    
    #def compute_tree(self,minpatches=2,epsilon=1.e-9):
    #    tovisit = queue.Queue()
    #    tovisit.put(self)
    #    dictelements = []
    #    self.leafs = []
    #    cur_nodes = set()
    #    prev_nodes = set()
    #    prev_nodes.add(self)
    #    depth = 0
    #    while not tovisit.empty():
    #        cur = tovisit.get()
    #        c1,c2 = cur.branch(epsilon,minpatches)
    #        if c1 is None and c2 is None:
    #            #in this case do not branch. we arrived at a leaf
    #            self.leafs += [cur]
    #            continue
    #        if cur in prev_nodes:
    #            cur_nodes.add(c1)
    #            cur_nodes.add(c2)
    #        else:
    #            prev_nodes = cur_nodes
    #            cur_nodes = set()
    #            depth += 1
    #        centroid1 = sum([p.matrix for p in c1.patches])
    #        centroid2 = sum([p.matrix for p in c2.patches])
    #        curdict = centroid1/c1.npatches - centroid2/c2.npatches
    #        if np.linalg.norm(curdict) < 1.e-3:
    #            ipdb.set_trace()
    #        dictelements += [curdict]
    #        tovisit.put(c1)
    #        tovisit.put(c2)
    #    self.dictelements = dictelements
    #    self.depth = depth
    #    return(dictelements)
    #    
    #def branch(self,epsilon,minpatches):
    #    if self.children is not None:
    #        raise Exception("Node already has children")
    #    n1,n2 = None,None
    #    if self.npatches > minpatches:
    #        sep_idx,val,centroid_dist = oneDtwomeans([p.eigid for p in self.patches])
    #        if centroid_dist > epsilon:
    #            p1,p2 = self.patches[:sep_idx],self.patches[sep_idx:]
    #            n1,n2 = Node(p1,self),Node(p2,self)
    #            self.children = [n1,n2]
    #    return(n1,n2)
