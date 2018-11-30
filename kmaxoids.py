import ipdb
import numpy as np
import itertools


class Kmaxoids():

    def __init__(self, Y, K=2):
        """KMaxoids class. 

        Y: matrix where each column represents a data point
        K: number of desired clusters"""
        
        self.Y = Y
        self.dim,self.nsamples = Y.shape
        self.K = K

    def run(self,nruns=10,maxit=20):
        """Runs the Lloyd's algorithm variant described in [1] multiple times, each time with different random choices of initial maxoids. The final clusters are given by the run that gives best value of the optimization function.

        nruns: the number of times the algorithm is run 
        maxit: the number of iterations for each run

        [1]: http://ceur-ws.org/Vol-1458/E19_CRC4_Bauckhage.pdf"""

        best_val = np.infty
        for i in range(nruns):
            maxoids,clusters = self._run_once(maxit)
            val = 0
            for k,colnum in itertools.product(range(self.K),range(self.nsamples)):
                x = self.Y[:,colnum]
                m = maxoids[:,k]
                val += np.linalg.norm(x-m)**2
            if val < best_val:
                best_val = val
                best_max,best_clust = maxoids,clusters
                print(best_val)
        return(best_max,best_clust)
            
    def _run_once(self,maxit):
        """Runs the algorithm only once and returns the clusters"""

        maxoids = self.Y[:,np.random.choice(self.nsamples,size=self.K,replace=False)]
        #print(maxoids)
        for i in range(maxit):
            #Update Clusters
            clusters = np.array([set() for k in range(self.K)])
            for colnum,dpoint in enumerate(self.Y.transpose()):
                #dist = np.linalg.norm(dpoint - self.Y[:,colnum+1 if colnum + 1 < self.nsamples else colnum - 1])
                #dist = np.linalg.norm(dpoint - maxoids[:,0])
                dist = np.infty
                for maxn,m in enumerate(maxoids.transpose()):
                    cand_dist = np.linalg.norm(dpoint - m)
                    if cand_dist < dist:
                        dist = cand_dist
                        max_idx = maxn
                clusters[max_idx].add(colnum)

            #update maxoids
            for clustn,clust in enumerate(clusters):
                sum_dist = -np.infty
                for colnum in clust:
                    x = self.Y[:,colnum]
                    cand_sum_dist = np.sum([np.linalg.norm(x - m) for m in maxoids.transpose()])
                    if cand_sum_dist > sum_dist:
                        sum_dist = cand_sum_dist
                        maxoids[:,clustn] = x
        return(maxoids,clusters)
