import random
import math
import numpy
from numpy import matrix as MA
from knnsearch import knnsearch
from build_graph import build_graph
from randd import randd
def  search_graph(query, nodes, DS, K):
    dt = numpy.dtype('f8')
    random.seed(100)
    k = nodes.shape[1]
    depth = 0
    flag  = 0
    parent_id = randd(1, nodes.shape[0], 1); # Check this
    visited = 1;
    while 1:
        parent_vec = MA(DS[parent_id,0:],dt)  # parent node
        child_ids = nodes[parent_id,0:];
        Val = numpy.zeros((len(child_ids),DS.shape[1]),dt)
        I = 0 
        while I < len(child_ids):
            Val[I,0:] = DS[child_ids[I],0:]
            I += 1
        (nn1_ind, nn1_dist) = knnsearch(query,Val, K)
        visited = visited + k
        if (parent_dist <= nn1_dist):
            flag=1
            break
        parent_id = child_ids(nn1_ind)
        depth = depth+1
    if flag == 1:
        nn_id = parent_id
        nn_dist  = parent_dist
    else:
        nn_id = - 1
        nn_dist = -1

    return nn_id, nn_dist, visited

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
    nodes = build_graph(X,3)
    Q = [random.randint(0,100) for r in range(10)]
    print Q
    Q = MA(Q,dt)
    print (search_graph(Q,nodes,X,1))
    
