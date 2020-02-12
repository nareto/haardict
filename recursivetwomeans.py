
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
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import queue
   
class Node():
    def __init__(self,
                 samples_idx,
                 parent,
                 isleftchild=True,
                 representative_data_point=None):
        
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
        self.samples_idx = np.array(samples_idx) #indexes of d.samples_idx where d is an ocdict
        self.nsamples= len(self.samples_idx)
        self.representative_data_point = representative_data_point

    def __gt__(self, other):
        return self.depth > other.depth

    def __lt__(self, other):
        return other.depth > self.depth


class RecursiveTwoMeans(BaseEstimator, TransformerMixin):

    def __init__(self, wcss_tol = 1, ncomponents = None):
        """wcss_tol: the upper bound on all the 'final' clusters (corresponding to leafs in the dendogram).
        ncomponents: If set the PriorityQueue version of the method is used to produce this number of clusters. In this case wcss_tol is ignored."""

        self.wcss_tol = wcss_tol
        self.ncomponents = ncomponents
        
        if self.ncomponents is not None and type(self.ncomponents) != type(1):
            raise Exception("ncomponents must be an int")

        if self.ncomponents is not None:
            self.visit = 'priority'
        else:
            self.visit = 'fifo'
        self.leafs = []

    def _split_set(self, X, data_indexes):
        """Returns (idx1,idx2,lcentroid, rcentroid, wcss) where idx1 and idx2 are the indexes of the split with centroids lcentroid and rcentroid; wcss is the achieved value of the WCSS function.

        X: the whole data matrix
        data_indexes: the subset of indexes to split with 2-means"""

        km_instance = KMeans(n_clusters=2,tol=0.1)
        km = km_instance.fit(X[data_indexes,:])
        #self.wcss.append(km.inertia_)
        lsamples = (km.labels_ == 0).nonzero()[0]
        rsamples = (km.labels_ != 0).nonzero()[0]
        #scaledwcss = km.inertia_/(len(data_indexes*self.data_dim))
        wcss = km.inertia_
        lcentroid = km.cluster_centers_[0,:]
        rcentroid = km.cluster_centers_[1,:]
        return(lsamples,rsamples,lcentroid,rcentroid,wcss)

    def fit(self, X, minsamples = 5):
        """Compute the clustering.

        minsamples: minimum number of samples to be in any to-be-split tree-node."""
        n_samples, n_features = X.shape

        nclusters = 1

        self.root_node = Node(tuple(np.arange(n_samples)),None)
        if self.visit == 'priority':
            tovisit = queue.PriorityQueue()
            totvar = X.var()
            tovisit.put((0,self.root_node))
        elif self.visit == 'fifo':
            tovisit = queue.Queue()
            tovisit.put((None,self.root_node))
        depth = 0 
        while not tovisit.empty():
            father_priority,cur = tovisit.get() #it always holds: father_priority = totwcss - wcss(cur)
            if cur.nsamples > minsamples and (self.visit != 'priority' or nclusters < self.ncomponents):
                curlsamples, currsamples, lrepr, rrepr, clust_ret_val = self._split_set(X, cur.samples_idx)
                #has_sons = True if curlsamples is not None and currsamples is not None else False
                minsize = min(len(curlsamples),len(currsamples))
                if minsize > 0 and (self.visit == 'priority' or clust_ret_val > self.wcss_tol): #decide whether or not to branch
                    abs_lsamples = cur.samples_idx[curlsamples]
                    abs_rsamples = cur.samples_idx[currsamples]
                    if self.visit == 'priority':
                        lvar = X[abs_lsamples,:].var()
                        rvar = X[abs_rsamples,:].var()                        
                        lpriority = totvar - lvar
                        rpriority = totvar - rvar
                        nclusters += 1
                    else:
                        lpriority = None
                        rpriority = np.finfo(float).eps
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
        self.labels_ = - np.ones(n_samples, dtype=int)
        self.cluster_centers_ = np.zeros((n_samples,n_features))
        for i,leaf in enumerate(self.leafs):
            self.labels_[leaf.samples_idx] = int(i)
            self.cluster_centers_[i,:] = leaf.representative_data_point
        return self
        
    def fit_predict(self, X, y = None):
        """Compute the clustering and predict cluster index for each sample"""

        self.fit(X,y)
        self.predict(X,y)

    def fit_transform(self, X, y = None):
        """Compute clustering and transform X to cluster-distance space."""

        self.fit(X,y)
        self.transform(X,y)
        
    def predict(self, X, y = None):
        pass

    def transform(self, X, y = None):
        pass

 
