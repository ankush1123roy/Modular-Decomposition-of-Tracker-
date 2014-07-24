
from knnsearch import knnsearch
import random
def  build_graph(X, k):

    ''' Build a Connected graph using k neighbours
        Author: Ankush Roy (ankush2@ualberta.ca)
                Kiana Hajebi (hajebi@ualberta.ca)
    '''
    cdef int I = 0
    cdef int i = 0 
    cdef int nodes[5000]
    cdef int nns_inds[41]
    cdef double nns_dists[41]
    cdef double query[1][1600]
    
    for i in range(X.shape[0]):
        for j in range(1600):
            query[0][j] = X[i][j]
        
        (nns_inds, nns_dists) = knnsearch(query,X,k+1)
        while I < len(nns_inds):
            if nns_inds[I] == i-1:
                nns_inds.remove(i-1)
                nodes[i-1,0:] = nns_inds
                break            
            else:
                I += 1
    return nodes
