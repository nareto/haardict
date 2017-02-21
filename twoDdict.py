import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as sslinalg
import skimage.io
import gc
import queue

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
    val = 0
    for s in C1:
        val += (s-c1bar)**2
    for s in C2:
        val += (s-c2bar)**2
    return(val)


def is_sorted(values):
    prev_v = values[0]
    for v in values[1:]:
        if v < prev_v:
            return(False)
    return(True)

def oneDtwomeans(values):
    best_kval = None
    best_idx = None
    prev_val = values[0]
    for separating_idx in range(1,len(values)):
        val = values[separating_idx]
        if val < prev_val:
            raise Exception("Input list must be sorted")
        prev_val = val
        kval = twomeansval(values,separating_idx)
        if best_kval is None or kval < best_kval:
            best_kval = kval
            best_idx = separating_idx
    return(best_idx,best_kval)
    

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

def main():
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
    for f in images_paths[:2]:
        images.append(skimage.io.imread(f,as_grey=True))
        
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
    dictelems = root_node.compute_tree()
    return(root_node,dictelems)

class Patch():
    def __init__(self,array,v1,v2):
        self.array = array
        self.v1 = v1
        self.v2 = v2
        self.eigid = float(np.dot(np.dot(v1.transpose(),array),v2))

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
        
    def compute_tree(self,minpatches=2):
        tovisit = queue.Queue()
        tovisit.put(self)
        dictelements = []
        self.leafs = []
        while not tovisit.empty():
            cur = tovisit.get()
            if cur.npatches <= minpatches:
                self.leafs += [cur]
                continue
            c1,c2 = cur.branch()
            centroid1 = sum([p.array for p in c1.patches])
            centroid2 = sum([p.array for p in c2.patches])
            curdict = centroid1/c1.npatches - centroid2/c2.npatches
            dictelements += [curdict]
            tovisit.put(c1)
            tovisit.put(c2)
        self.dictelements = dictelements
        return(dictelements)
        
    def branch(self):
        if self.children is not None:
            raise Exception("Node already has children")
        sep_idx,val = oneDtwomeans([p.eigid for p in self.patches])
        p1,p2 = self.patches[:sep_idx],self.patches[sep_idx:]
        n1,n2 = Node(p1,self),Node(p2,self)
        self.children = [n1,n2]
        return(n1,n2)

    

    
    
    
