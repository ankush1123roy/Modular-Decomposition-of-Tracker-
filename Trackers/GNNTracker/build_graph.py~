""" Original Matlab Code by Kiana Hajebi 
Changed to Python by Ankush Roy. This works for K = 1"""
from knnsearch import knnsearch
import random
import numpy
from numpy import matrix as MA
def  build_graph(X, k):
    dt = numpy.dtype('f8')
    f=[]
    nodes = numpy.zeros((X.shape[1],k),dt);
    i = 0
    while i <= X.shape[0]:
#        print i
        query = MA(X[(i-1),0:],dt)
        (nns_inds, nns_dists) = knnsearch(query,X,k+1)
        I  = 0 
        f = []
#        import pdb;pdb.set_trace()
        while I < len(nns_inds):
            if nns_inds[I] == i-1:
                nns_inds.remove(i-1)
                print i
                nodes[i-1,0:] = nns_inds
                break            
            else:
                I += 1
        i += 1
    print nodes
    return nodes

if __name__ == '__main__':
    dt = numpy.dtype('f8')
    X = [range(10)]
    print X
    for i in range(10):
        Vect = [random.randint(0,100) for r in range(10)]
        X = numpy.vstack([X,Vect])
    X = numpy.delete(X,(0),axis = 0)
    X = MA(X,dt)
    K = 2
    print X
    build_graph(X,K)
