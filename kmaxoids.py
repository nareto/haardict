import ipdb
import numpy as np
import itertools


class KMaxoids():
    def __init__(self, Y, K=2):
        """KMaxoids class. 

        Y: matrix where each column represents a data point
        K: number of desired clusters"""
        
        self.Y = Y
        self.dim,self.nsamples = Y.shape
        self.K = K

    def run(self,nruns=3,maxit=20):
        """Runs the Lloyd's algorithm variant described in [1] multiple times, each time with different random choices of initial maxoids. The final clusters are given by the run that gives best value of the optimization function. Returns tuple (maxoids,clusters) where:
        - maxoids is a matrix where each column is the reprensetative of the cluster
        - clusters is an array where each element is a set containing the index (column of self.Y) corresponding to the data point in that cluster

        Arguments:
        nruns: the number of times the algorithm is run 
        maxit: the number of iterations for each run

        [1]: http://ceur-ws.org/Vol-1458/E19_CRC4_Bauckhage.pdf"""

        best_val = np.infty
        for i in range(nruns):
            maxoids,labels = self._run_once(maxit)
            val = np.sum((self.Y - maxoids[:,labels])**2)
            #val = 0
            #for colnum,dpoint in self.Y.transpose():
            #    val += np.linalg.norm(dpoint - maxoids[:,labels[colnum]])**2
            if val < best_val:
                best_val = val
                best_max,best_labels = maxoids,labels
        self.val = best_val
        self.maxoids = best_max
        self.labels = best_labels
        return(best_max,best_labels)

    def _run_once(self,maxit):
        """Runs the algorithm only once and returns the clusters"""

        maxoids = self.Y[:,np.random.choice(self.nsamples,size=self.K,replace=False)]
        Y = self.Y.transpose()
        for i in range(maxit):
            #Update Clusters
            D = np.hstack([np.sum((Y-m)**2,axis=1).reshape(self.nsamples,1) for m in maxoids.transpose()])
            labels = np.argmin(D,axis=1)
            
            #update maxoids
            for k in range(self.K):
                dat_mat = self.Y[:,labels == k]
                max_mat = np.hstack([maxoids[:,0:k],maxoids[:,k+1:]]).transpose()
                sum_dist = -np.infty
                for dpoint in dat_mat.transpose():
                    dist = np.sum((max_mat - dpoint)**2)
                    if dist > sum_dist:
                        maxoids[:,k] = dpoint
                        sum_dist = dist

        return(maxoids,labels)
