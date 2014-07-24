""" Original Matlab code by Kiana changed by Ankush """

import random
import math
import numpy 
from numpy import matrix as MA
def randd(lb, up, number):
    ind = []
    up=up+1 # [lb,ub]
    if number>=up:
        print('number should be <= up+1')
        return

    a,b=lb,up
    n=1
    ind = math.floor(a + (b-a)* random.random())
    while n < number:
        r = floor(a + (b-a)*random.random())
        if  isempty(find(ind==r)):  #any(ind==r):
            ind = numpy.hstack([ind,r])
            n=n+1
    print ind
    return ind
