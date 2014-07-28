"""
This module contains functions used in the particle filter
based tracking algorithms.

Based on parts including l1tracker, ivt, nntracker.
Author:  Xi Zhang
"""

from numpy import linalg
import numpy
import numpy as np
from numpy import matrix as MA
import pdb
import pylab
#from scipy import interpolate
#from scipy import weave
#from scipy.weave import converters
import cython
from cython.parallel cimport prange
import time

cdef extern from "math.h": 
    double floor(double)  
    double ceil(double) 
    double sqrt(double)

'''
def affparaminv(est):
    #!!!pay attention
    dt = np.dtype('f8')
    q = MA(np.zeros(2,3),dt)
    q = linalg.pinv(MA([[est[0,2],est[0,3]],[est[0,4],est[0,5]]])) * MA([[-est[0,0],1.0,0.0],[-est[0,1],0,1.0]])
    q=q.flatten(1)
    return MA([[q[0,0],q[0,1],q[0,2],q[0,4],q[0,3],q[0,5]]])
'''



def affparam2geom(est):
    # !!!pay attention
    dt = np.dtype('f8')
    A = MA([[est[0,2],est[0,3]],[est[0,4],est[0,5]]])
    U,S,V = linalg.svd(A,full_matrices=True)
    temp = MA(np.zeros((2,2),dt))
    #temp[0,0] = S[0]
    #temp[1,1] = S[1]
    #S = temp
    #import pdb; pdb.set_trace()
    if(linalg.det(U) < 0):
        U = U[:,range(1,-1,-1)]
        V = V[:,range(1,-1,-1)]
        S = S[:,range(1,-1,-1)]
        temp[1,1] = S[1]
        temp[0,0] = S[0]
        S = temp
    else:
        temp[1,1] = S[0]
        temp[0,0] = S[1]
        S = temp	
    #import pdb; pdb.set_trace()
    q = MA(np.zeros((1,6)),dt)
    q[0,0] = est[0,0]
    q[0,1] = est[0,1]
    q[0,3] = np.arctan2(U[1,0]*V[0,0]+U[1,1]*V[0,1],U[0,0]*V[0,0]+U[0,1]*V[0,1])
    phi = np.arctan2(V[0,1],V[0,0])
    if phi <= -np.pi/2:
        c = np.cos(-np.pi/2)
        s = np.sin(-np.pi/2)
        R = MA([[c,-s],[s,c]])
        V = MA(V) * MA(R)
        S = R.T*MA(S)*R
    if phi > np.pi/2:
        c = np.cos(np.pi/2)
        s = np.sin(np.pi/2)
        R = MA([[c,-s],[s,c]])
        V = MA(V)*MA(R)
        S = R.T*MA(S)*R
    #import pdb; pdb.set_trace()
    q[0,2] = S[0,0]
    q[0,4] = S[1,1]/S[0,0]
    q[0,5] = np.arctan2(V[0,1],V[0,0])
    return q

cdef double[:,:] warpimg(double[:,:] img, double[:,:] p, double[:,:] sz):
    # arrays
    cdef double w,h
    cdef double[:] x
    cdef double[:] y
    cdef double[:,:] result
    #cdef double[:,:] xcoord
    #cdef double[:,:] ycoord
    cdef double[:,:] temp
    cdef double[:] temp1
    w = sz[0,0]
    h = sz[0,1]
    result = np.zeros((w*h,p.shape[0]),dtype=np.float64)
    #May have problem    
    x = np.linspace(1-h/2,h/2,h)
    y = np.linspace(1-w/2,w/2,w)
    xv,yv = np.meshgrid(x,y)
    for i in xrange(p.shape[0]):
        temp = np.concatenate((np.ones((w*h,1)),xv.reshape((w*h,1),order='F'),yv.reshape((w*h,1),order='F')),axis=1)*MA([[p[i,0],p[i,1]],[p[i,2],p[i,4]],[p[i,3],p[i,5]]])
        temp1 = sample_region(img, np.array(temp))
        result[:,i] = temp1[:]
    return result

# TODO
cdef double bilinear_interp(double [:,:] img, double x, double y):
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]

    cdef unsigned int lx = <int>floor(x)
    cdef unsigned int ux = <int>ceil(x)
    cdef unsigned int ly = <int>floor(y)
    cdef unsigned int uy = <int>ceil(y)

    # Need to be a bit careful here due to overflows
    if not (0 <= lx < w and 0 <= ux < w and
            0 <= ly < h and 0 <= uy < h): return 128

    cdef double dx = x - lx
    cdef double dy = y - ly
    return img[ly,lx]*(1-dx)*(1-dy) + \
           img[ly,ux]*dx*(1-dy) + \
           img[uy,lx]*(1-dx)*dy + \
           img[uy,ux]*dx*dy

cdef double[:] sample_region(double[:,:] img, double[:,:] temp):
    cdef int num_pts
    cdef int width
    cdef int height
    cdef int i,j,k
    cdef double w,h
    cdef double[:] result

    num_pts = temp.shape[0]
    height = img.shape[0]    
    width = img.shape[1]
    result = np.empty(num_pts, dtype=np.float64)
    j = 0
    k = 1
    for i in xrange(num_pts):
        w = temp[i,j]
        h = temp[i,k]
        result[i] = bilinear_interp(img, w, h)
    return result
                

def affparam2mat(p):
    # Here I use array instead of matrix
#    cdef double[:] s,th,r,phi
#    cdef double[:] cth,sth,cph,sph
#    cdef double[:] ccc,ccs,css,scc,scs,sss
#    cdef double[:,:]  q
    q = np.zeros(np.asarray(p).shape,'f8')
    #import pdb; pdb.set_trace()
    #if len(p.shape) == 1: 
#	temp = np.zeros(1,p.shape[0])
#	temp[0,:] = p
    s = np.array(p[:,2])
    th = np.array(p[:,3])
    r = np.array(p[:,4])
    phi = np.array(p[:,5])
    cth = np.cos(th)
    sth = np.sin(th)
    cph = np.cos(phi)
    sph = np.sin(phi)
    ccc = cth*cph*cph
    ccs = cth*cph*sph
    css = cth*sph*sph
    scc = sth*cph*cph
    scs = sth*cph*sph
    sss = sth*sph*sph
    q[:,0] = np.array(p[:,0])
    q[:,1] = np.array(p[:,1])
    q[:,2] = s*(ccc +scs +r*(css -scs))
    q[:,3] = s*(r*(ccs -scc) -ccs -sss)
    q[:,4] = s*(scc -ccs +r*(ccs +sss))
    q[:,5] = s*(r*(ccc +scs) -scs +css)
    return MA(q)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double[:] est_weight_warp(double[:,:] img, double[:,:] template, double[:,:] param, int n_samples, double[:,:] sz_T, double[:,:] affsig, double alpha):
    cdef int n_process, i, partial
    cdef double[:] wlist
    n_process = 2
    partial = n_samples / n_process
    #cdef double[:,:] temp_param
    wlist = np.empty(n_samples, dtype=np.float64)
    # Jesse: Try later
    #for i in prange(n_process, num_threads=n_process, nogil=True):
    #    wlist[i*partial:(i+1)*partial] = weight_process(img, template, param[i*partial:(i+1)*partial,:], partial, sz_T, affsig, alpha)
    return wlist
''' 
cdef double[:] weight_process(double[:,:] img, double[:,:] template, double[:,:] sub_param, int sub_n_samples, double[:,:] sz_T, double[:,:] affsig, double alpha) nogil:
    cdef double[:,:] sub_samples
    cdef double[:,:] new_param
    cdef double[:] sub_wlist
    new_param = sub_param + np.random.randn(sub_n_samples,6) * np.tile(np.array(affsig),[sub_n_sample,1]) 
    sub_samples =  warpimg(img,affparam2mat(new_param), sz_T)
    sub_wlist = weight_eval(sub_samples, template, sz_T, alpha)
    return sub_wlist
'''
def estwarp_condens(img,param,n_sample,sz_T,affsig):
        #again array
        #param = np.array(np.tile(affparam2geom(est),[n_sample,1]))
        #import pdb; pdb.set_trace()
        param = param + np.random.randn(n_sample,6) * np.tile(np.array(affsig),[n_sample,1])
        samples = warpimg(img,affparam2mat(param),sz_T)
        #import pdb; pdb.set_trace()
        return samples,param

def resample2(curr_samples,prob):
        dt = np.dtype('f8')
        nsamples = MA(curr_samples.shape).item(0)
        afnv = 0
        #pdb.set_trace()
        if prob.sum() ==0 :
                #import pdb; pdb.set_trace()
                map_afnv = MA(np.ones(nsamples),dt).T*afnv
                count = MA(np.zeros((prob.shape),dt))
        else:
                prob = prob/prob.sum()
                N = nsamples
                Ninv = 1 / float(N)
                map_afnv = MA(np.zeros((N,6)),dt)
                c = pylab.cumsum(prob)
                u = pylab.rand()*Ninv
                i = 0
                #pdb.set_trace()
                for j1 in range(N):
                        uj = u + Ninv*j1
                        while uj > c[i]:
                                i += 1
                        map_afnv[j1,:] = curr_samples[i,:]
                return map_afnv

cdef double [:] weight_eval(double[:,:] samples, double[:,:] template, double[:,:] sz,double alpha):
    #wlist = [i for i in range(samples.shape[1])]
    cdef int i
    cdef double wlist_u
    cdef double [:] wlist
    cdef double [:] temp_norm
    cdef double [:,:] temp_sample
    cdef double [:,:] temp_template
    wlist = np.zeros(samples.shape[1])
    a1 = time.time() 
    temp_samples  = whiten(samples)
    a2 = time.time()
    temp_samples = normalizeTemplates(temp_samples)
    a3 = time.time() 
    temp_template = np.asarray(template).reshape((sz[0,0]*sz[0,1],1),order='F')
    for i in xrange(temp_samples.shape[1]):
        temp_norm = np.array(temp_samples[:,i]) - np.array(temp_template[:,0])
        wlist_u = np.linalg.norm(temp_norm)
        wlist[i] = np.exp(-alpha*(wlist_u*wlist_u))
        if str(wlist[i]) == 'nan':pdb.set_trace()    
    #import pdb; pdb.set_trace()
    a4 = time.time()
    #print 'Jesse2'
    #print a2-a1
    #print a3-a2
    #print a4-a3
    return wlist

def drawbox(est,sz):
    temp = np.zeros(6)
    #import pdb;pdb.set_trace()
    temp[:] = est[0,:]
    est = temp
    #temp[:] = est[0,:]
    M = np.array([[est[0],est[2],est[3]],[est[1],est[4],est[5]]])
    w = sz[0,0]
    h = sz[0,1]
    corners = np.array([[1,-w/2,-h/2],[1,w/2,-h/2],[1,w/2,h/2],[1,-w/2,h/2]]).T
    corners = MA(M) * MA(corners)
    #import pdb; pdb.set_trace()
    return corners

cdef double[:,:] whiten(double[:,:] In):
    cdef int MN
    #cdef double [:] a,b
    cdef double[:,:] out
    # TODO change the type of a,b
    dt = np.dtype('f8')
    MN = np.asarray(In).shape[0]
    a = np.mean(In,axis=0)
    b = np.std(In,axis=0) + 1e-14
    #print b.shape
    out = np.divide( ( In - (MA(np.ones((MN,1)),dt)*a)), (MA(np.ones((MN,1)),dt) *b) )
    return out

def des_sort(q):
        temp = q
        B1 = numpy.sort(q);
        Asort = B1[::-1];
        B = numpy.argsort(q)
        A_ind= B[::-1]
        #pdb.set_trace()
        return Asort, A_ind

cdef double[:,:] normalizeTemplates(double[:,:] A):
        cdef int MN
        cdef double[:] A_norm
        MN = A.shape[0]
        A_norm = np.sqrt(np.sum(np.multiply(A,A),axis=0)) + 1e-14;
        A = np.divide(A,(np.ones((MN,1))*A_norm));
        return A
