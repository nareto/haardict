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
    if row_wise:
        m = centroid.shape[1]
    else:
        m = centroid.shape[0]
    ret = np.zeros(shape=(m,m))
    for patch in patches:
        addend = np.dot((patch - centroid).transpose(),(patch-centroid))
        ret += addend
    ret /= len(patches)
    return(ret)

class Patch():
    def __init__(self,matrix):
        self.matrix = matrix
        self.feature_matrix = None

    def compute_feature_matrix(self,U,V):
        self.feature_matrix = np.dot(np.dot(V.transpose(),self.matrix),U)
        
    def show(self):
        fig = plt.figure()
        axis = fig.gca()
        plt.imshow(self.array, cmap=plt.cm.gray,interpolation='none')
        fig.show()

class twodpca():
    def __init__(self,patches,l=-1,r=-1):
        self.l = l
        self.r = r
        self.patches = tuple(patches)
        self.bilateral_computed = False
        
    def compute_horizzontal_2dpca(self):
        cov = covariance_matrix([p.array for p in self.patches])
        eigenvalues,U = sslinalg.eigs(cov,l) 
        self.U = U.real
        
    def compute_vertical_2dpca(self):
        cov = covariance_matrix([p.array.transpose() for p in self.patches])
        eigenvalues,V = sslinalg.eigs(cov,r) 
        self.V = V.real
        
    def compute_simple_bilateral_2dpca(self):
        if self.bilateral_computed:
            raise Exception('Bilateral 2dpca already computed')
        self.compute_horizzontal_2dpca()
        self.compute_vertical_2dpca()
        self.bilateral_computed = True
    
    def compute_simple_bilateral_2dpca(self):
        if self.bilateral_computed:
            raise Exception('Bilateral 2dpca already computed')
        #TODO: implement

        

class ocdict():
    def __init__(self, patches):
        self.patches = tuple(patches)
        self.npatches = len(patches)
        #self.height,self.width = patches[0].shape
        self.shape = self.patches[0].shape
        #self.matrix = np.hstack([np.hstack(np.vsplit(x,self.height)).transpose() for x in self.patches])
        #self.normalization_coefficients = np.ones(shape=(self.matrix.shape[1],))
        #self.matrix_is_normalized = False
        #self.normalize_matrix()
        #self.sparse_coefs = None
        self.root_node = root_node

        
    def save_pickle(self,filepath):
        f = open(filepath,'wb')
        pickle.dump(self.__dict__,f,3)
        f.close()

    def load_pickle(self,filepath):
        f = open(filepath,'rb')
        tmpdict = pickle.load(f)
        f.close()
        self.__dict__.update(tmpdict)

    def normalize_matrix(self):
        self.normalized_matrix = np.copy(self.matrix)
        ncols = self.matrix.shape[1]
        for j in range(ncols):
            col = self.normalized_matrix[:,j]
            norm = np.linalg.norm(col)
            col /= norm
            self.normalization_coefficients[j] = norm
        self.matrix_is_normalized = True

    def slink_kmeans_cluster(self):
        pass

    def twomeans_cluster(self,minpatches=2,epsilon=1.e-2):
        self.root = Node(tuple(np.arange(self.npatches)),None)
        tovisit = queue.Queue()
        tovisit.put(self)
        #dictelements = []
        self.leafs = []
        cur_nodes = set()
        prev_nodes = set()
        prev_nodes.add(self)
        depth = 0
        patches_matrix = np.vstack([p.array.flatten() for p in self.patches]) 
        while not tovisit.empty():
            cur = tovisit.get()
            lpatches = []
            rpatches = []
            if cur.npatches > minpatches:
                km = KMeans(n_clusters=2).fit(patches_matrix[np.array(cur.patches)])
                if km.inertia_ > epsilon:
                    for idx,label in enumerate(km.labels_):
                        if label == 0:
                            lpatches.append(cur.patches[idx])
                        if label == 1:
                            rpatches.append(cur.patches[idx])
                    lnode = Node(lpatches,cur)
                    rnode = Node(rpatches,cur)
                    cur.children = (lnode,rnode)
                    tovisit.put(lnode)
                    tovisit.put(rnode)
            if len(lpatches) == 0 and len(rpatches) == 0:
                self.leafs.append(cur)
            #c1,c2 = cur.branch(epsilon,minpatches)
            #if c1 is None and c2 is None:
            #    #in this case do not branch. we arrived at a leaf
            #    self.leafs += [cur]
            #    continue
            #if cur in prev_nodes:
            #    cur_nodes.add(c1)
            #    cur_nodes.add(c2)
            #else:
            #    prev_nodes = cur_nodes
            #    cur_nodes = set()
            #    depth += 1
            #centroid1 = sum([p.array for p in c1.patches])
            #centroid2 = sum([p.array for p in c2.patches])
            #curdict = centroid1/c1.npatches - centroid2/c2.npatches
            #if np.linalg.norm(curdict) < 1.e-3:
            #    ipdb.set_trace()
            #dictelements += [curdict]
            #tovisit.put(c1)
            #tovisit.put(c2)
        self.dictelements = dictelements
        self.depth = depth
        return(dictelements)


        
    def sparse_code(self,input_patch,sparsity):
        if input_patch.array.shape != self.patches[0].shape:
            raise Exception("Input patch is not of the correct size")
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity,normalize=True)
        #omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity,fit_intercept=True,normalize=True,tol=1)
        #omp = OrthogonalMatchingPursuit(fit_intercept=True,normalize=True,tol=1)
        y = np.hstack(np.vsplit(input_patch.array,self.height)).transpose()
        if self.matrix_is_normalized:
            #omp.fit(self.normalized_matrix,y)
            #coef = omp.coef_
            coef = octave.OMP(sparsity,y.astype('float64').transpose(),self.normalized_matrix.astype('float64')).transpose()
            for idx,norm in np.ndenumerate(self.normalization_coefficients):
                coef[idx] /= norm
        else:
            omp.fit(self.matrix,y)
            coef = omp.coef_
            #coef = octave.OMP(sparsity,y.astype('float64').transpose(),self.matrix.astype('float64')).transpose()
        #print('Error in sparse coding: %f' % np.linalg.norm(input_patch.array - self.decode(coef)))
        return(coef)

    def decode(self,coefs):
        out = np.zeros_like(self.patches[0])
        for idx in coefs.nonzero()[0]:
            #out += coefs[idx]*self.patches[idx]
            out += coefs[idx]*self.matrix[:,idx].reshape(self.shape)
            #out += coefs[idx]*self.normalized_matrix[:,idx].reshape(self.shape)
        #if out.min() < 0:
        #    out -= out.min()
        #out /= out.max()
        return(out)
    
class Node():
    def __init__(self,patches,parent):
        self.parent = parent
        self.children = None
        self.patches = tuple(patches)
        self.npatches = len(self.patches)

    def build_patches_card_list(self):
        out = []
        for leaf in self.leafs:
            out.append(leaf.npatches)
        return(out)
        
        
    def compute_tree(self,minpatches=2,epsilon=1.e-9):
        tovisit = queue.Queue()
        tovisit.put(self)
        dictelements = []
        self.leafs = []
        cur_nodes = set()
        prev_nodes = set()
        prev_nodes.add(self)
        depth = 0
        while not tovisit.empty():
            cur = tovisit.get()
            c1,c2 = cur.branch(epsilon,minpatches)
            if c1 is None and c2 is None:
                #in this case do not branch. we arrived at a leaf
                self.leafs += [cur]
                continue
            if cur in prev_nodes:
                cur_nodes.add(c1)
                cur_nodes.add(c2)
            else:
                prev_nodes = cur_nodes
                cur_nodes = set()
                depth += 1
            centroid1 = sum([p.array for p in c1.patches])
            centroid2 = sum([p.array for p in c2.patches])
            curdict = centroid1/c1.npatches - centroid2/c2.npatches
            if np.linalg.norm(curdict) < 1.e-3:
                ipdb.set_trace()
            dictelements += [curdict]
            tovisit.put(c1)
            tovisit.put(c2)
        self.dictelements = dictelements
        self.depth = depth
        return(dictelements)
        
    def branch(self,epsilon,minpatches):
        if self.children is not None:
            raise Exception("Node already has children")
        n1,n2 = None,None
        if self.npatches > minpatches:
            sep_idx,val,centroid_dist = oneDtwomeans([p.eigid for p in self.patches])
            if centroid_dist > epsilon:
                p1,p2 = self.patches[:sep_idx],self.patches[sep_idx:]
                n1,n2 = Node(p1,self),Node(p2,self)
                self.children = [n1,n2]
        return(n1,n2)
