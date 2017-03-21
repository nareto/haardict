import ipdb
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as sslinalg
import skimage.io
from sklearn.linear_model import OrthogonalMatchingPursuit
from oct2py import octave
import gc
import queue

def stack(mat):
    out = np.hstack(np.vsplit(mat,mat.shape[0])).transpose()
    return(out)

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
    
def covariance_matrix(patches, row_wise=False):
    centroid = sum(patches)/len(patches)
    if row_wise:
        m = centroid.shape[1]
    else:
        m = centroid.shape[0]
    ret = np.zeros(shape=(m,m))
    for patch in patches:
        if row_wise:
            addend = np.dot((patch - centroid).transpose(),(patch-centroid))
        else:
            addend = np.dot((patch - centroid),(patch-centroid).transpose())
        ret += addend
    ret /= len(patches)
    return(ret)

def learn_dict():
    images = []
    images_paths = []
    #for i in range(1):
    #    images.append(np.random.uniform(size=(200,300)))
    #starting_path = '/Users/renato/ownCloud/phd/bonin_et_al_pics_2009'
    starting_path = '/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Nature'
    for dirpath,dirnames,filenames in os.walk(starting_path):
        for f in filenames:
            if f[-3:].upper() in  ['JPG','GIF','PNG']:
                images_paths.append(dirpath+'/'+f)
    #paths = images_paths[:2]
    #paths = ['/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Nature/cliff.jpg','/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Nature/island.jpg']
    paths = ['/Users/renato/ownCloud/phd/360_Colour_Items_1Moreno-Martinez_Montoro/Nature/cliff.jpg']
    for f in paths:
        images.append(skimage.io.imread(f,as_grey=True))
    print('Learning from images: %s' % paths)
    tmp_patches = []
    for i in images:
        tmp_patches += extract_patches(i)
    G1 = covariance_matrix(tmp_patches)
    G2 = covariance_matrix(tmp_patches,True)
    l1,v1 = sslinalg.eigs(G1,1)
    l2,v2 = sslinalg.eigs(G2,1)
    v1 = v1.real
    v2 = v2.real
    patches = []
    for p in tmp_patches:
        patches.append(Patch(p,v1,v2))
    tmp_patches = None
    gc.collect()
    patches.sort(key=lambda p: p.eigid)
    root_node = Node(patches,None)
    dictelems = ocDict(root_node.compute_tree(10))
    return(root_node,dictelems)

def fast_test_reconstruction(sparsity=40):
    ocdict = ocDict()
    ocdict.load_pickle('/Users/renato/ownCloud/phd/code/2ddict/ocdict-MMM-Nature-2')
    #ocdict.load_pickle('/Users/renato/ownCloud/phd/code/2ddict/ocdict-MMM-Nature-2-nonnormalized')
    mountain = skimage.io.imread('/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Nature/mountain.jpg',as_grey=True)
    mountain_patches = extract_patches(mountain)
    p1 = Patch(mountain_patches[3455])
    test_reconstruction(ocdict,p1,sparsity=sparsity)


def test_reconstruction(ocdict,patch=None,sparsity = 40):
    if patch is None:
        patch = Patch(np.random.uniform(size=ocdict.shape))
    coeffs = ocdict.sparse_code(patch,sparsity)
    outpatch = ocdict.decode(coeffs)
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(patch.array, cmap=plt.cm.gray,interpolation='none')
    ax2.imshow(outpatch, cmap=plt.cm.gray,interpolation='none')
    ax1.set_title = 'original'
    ax2.set_title = 'reconstructed'
    fig.show()
    return(coeffs)

def test_reconstruction_patches(ocdict,patches=None,sparsity = 10):
    height,width = ocdict.shape
    if patches is None:
        patches = [Patch(np.random.uniform(size=ocdict.shape)),Patch(np.random.uniform(size=ocdict.shape))]
    height *= len(patches)
    img = np.vstack([x.array for x in patches])
    #patch = Patch(np.hstack([x.array for x in patches]))
    outpatches = []
    for patch in patches:
        outpatch = ocdict.decode(ocdict.sparse_code(patch,sparsity))
        outpatches.append(outpatch)
    result = assemble_patches(outpatches,(height,width))
    #fig,(ax1,ax2) = plt.subplots(1,2)
    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(img, cmap=plt.cm.gray,interpolation='none')
    ax2.imshow(result, cmap=plt.cm.gray,interpolation='none')
    #ax2.imshow(outpatches[0], cmap=plt.cm.gray,interpolation='none')
    ax3.imshow(outpatches[1], cmap=plt.cm.gray,interpolation='none')
    ax1.set_title = 'original'
    ax2.set_title = 'reconstructed'
    fig.show()

    
def test_denoise(ocdict):
    sigma_noise = 12
    psize = (ocdict.height,ocdict.width)
    sparsity = 20

    #imgpath =  '/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Living_Things/Marine_creatures/lobster.jpg'
    #imgpath =  '/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Nature/mountain.jpg'
    imgpath =  '/Users/renato/ownCloud/phd/code/epwt/img/cameraman256.png'
    img = skimage.io.imread(imgpath,as_grey=True)
    imgn = img + np.random.normal(0,sigma_noise,size=img.shape)
    hpi1 = HaarPSI(255*img,255*imgn)
    print('hpi1 = %f' % hpi1)
    patches = extract_patches(imgn,size=psize)
    outpatches = []
    for p in patches:
        outpatches.append(ocdict.decode(ocdict.sparse_code(Patch(p),sparsity)))
    out = assemble_patches(outpatches,img.shape)
    hpi2 = HaarPSI(255*img,255*out)
    print('hpi2 = %f' % hpi2)
    fig, (ax1, ax2,ax3) = plt.subplots(1, 3)#, sharey=True)
    ax1.imshow(img[34:80,95:137], cmap=plt.cm.gray,interpolation='none')
    ax2.imshow(imgn[34:80,95:137], cmap=plt.cm.gray,interpolation='none')
    ax3.imshow(out[34:80,95:137], cmap=plt.cm.gray,interpolation='none')
    ax1.set_title = 'original'
    ax2.set_title = 'noisy'
    ax3.set_title = 'denoised'
    fig.show()
    return(img,imgn,out)

def test_assembling():
    imgpath =  '/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Living_Things/Marine_creatures/lobster.jpg'
    img = skimage.io.imread(imgpath,as_grey=True)
    patches = extract_patches(img,size=(8,8))
    out = assemble_patches(patches,img.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(img, cmap=plt.cm.gray,interpolation='none')
    ax2.imshow(out, cmap=plt.cm.gray,interpolation='none')
    ax1.set_title = 'original'
    ax2.set_title = 're assembled'
    fig.show()

class Patch():
    def __init__(self,array,v1=None,v2=None):
        self.array = array
        self.v1 = v1
        self.v2 = v2
        if v1 is not None and v2 is not None:
            self.eigid = float(np.dot(np.dot(v1.transpose(),array),v2))

    def show(self):
        fig = plt.figure()
        axis = fig.gca()
        plt.imshow(self.array, cmap=plt.cm.gray,interpolation='none')
        fig.show()

   #def __add__(self, other_patch):
   #    if self.v1 != other_patch.v1 or self.v2 != other_patch.v2:
   #        raise Exception("You shouldn't do this")
   #    return(Patch(self.array+other_patch.array,self.v1,self.v2))
        
class Node():
    def __init__(self,patches,parent):
        self.parent = parent
        self.children = None
        self.patches = tuple(patches)
        self.npatches = len(self.patches)
        
    def save_pickle(self,filepath):
        f = open(filepath,'wb')
        pickle.dump(self.__dict__,f,3)
        f.close()

    def load_pickle(self,filepath):
        f = open(filepath,'rb')
        tmpdict = pickle.load(f)
        f.close()
        self.__dict__.update(tmpdict)

        
    def compute_tree(self,minpatches=2):
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
            c1,c2 = cur.branch()
            if cur.npatches <= minpatches or (c1 is None and c2 is None):
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
        
    def branch(self):
        if self.children is not None:
            raise Exception("Node already has children")
        sep_idx,val,centroid_dist = oneDtwomeans([p.eigid for p in self.patches])
        if centroid_dist > 1.e-9:
            p1,p2 = self.patches[:sep_idx],self.patches[sep_idx:]
            n1,n2 = Node(p1,self),Node(p2,self)
            self.children = [n1,n2]
        else:
            n1,n2 = None,None
        return(n1,n2)

class ocDict():
    def __init__(self, patches=None):
        if patches is not None:
            self.patches = tuple(patches)
            self.npatches = len(patches)
            self.height,self.width = patches[0].shape
            self.matrix = np.hstack([np.hstack(np.vsplit(x,self.height)).transpose() for x in self.patches])
            self.normalization_coefficients = np.ones(shape=(self.matrix.shape[1],))
            self.matrix_is_normalized = False
            self.normalize_matrix()
            self.shape = self.patches[0].shape
        self.sparse_coefs = None
        
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
        
    def sparse_code(self,input_patch,sparsity):
        if input_patch.array.shape != self.patches[0].shape:
            raise Exception("Input patch is not of the correct size")
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity,normalize=True)
        #omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity,fit_intercept=True,normalize=True,tol=1)
        #omp = OrthogonalMatchingPursuit(fit_intercept=True,normalize=True,tol=1)
        y = np.hstack(np.vsplit(input_patch.array,self.height)).transpose()
        if self.matrix_is_normalized:
            omp.fit(self.normalized_matrix,y)
            coef = omp.coef_
            #coef = octave.OMP(sparsity,y.astype('float64').transpose(),self.normalized_matrix.astype('float64')).transpose()
            for idx,norm in np.ndenumerate(self.normalization_coefficients):
                coef[idx] *= norm
        else:
            omp.fit(self.matrix,y)
            coef = omp.coef_
            #coef = octave.OMP(sparsity,y.astype('float64').transpose(),self.matrix.astype('float64')).transpose()
        #print('Error in sparse coding: %f' % np.linalg.norm(input_patch.array - self.decode(coef)))
        return(coef)

    def decode(self,coefs):
        out = np.zeros_like(self.patches[0])
        for idx in coefs.nonzero()[0]:
            out += coefs[idx]*self.patches[idx]
        #if out.min() < 0:
        #    out -= out.min()
        #out /= out.max()
        return(out)
    
