"""
Implementation of the simple PF Tracker.
In the framework of travis's NNtracker
Author: Xi Zhang
"""

import cv2

cimport numpy as np
import numpy as np

#from nntracker.utility import apply_to_pts_all, square_to_corners_warp, apply_to_pts_raw, normCrossCorrelation, robust_f
#from nntracker.utility cimport *
from nntracker.imgtool import *
from nntracker.imgtool cimport *
import time
import pdb

cdef class PFTracker:

    cdef:
        int n_samples
        int resx
        int resy
        np.ndarray template
        double alpha
        double threshold
        
        double[:,:] sz_T
        double[:,:] current_warp
        double[:] intensity_map
        bint use_scv
        bint initialized

        double[:,:] est
        double[:,:] affsig
        double[:,:] param

        double[:,:] init_img
        double[:,:] init_x_pos
        double[:,:] init_y_pos
        double[:,:] tmplt_size
        double[:,:] x_pos
        double[:,:] y_pos
        double[:,:] mask
        double[:] ncc

        int MModel
        int Algo
        dict options      

    def __init__(self, int n_samples, int resx, int resy, bint use_scv):
        self.n_samples = n_samples
        self.resx = resx
        self.resy = resy
        self.sz_T = np.empty((1,2),dtype=np.float64)
        self.sz_T[0,0] = self.resx
        self.sz_T[0,1] = self.resy

        self.affsig = np.array([[5.,5.,0.,0.02,0.02,0.001]],dtype=np.float64)
        self.alpha = 50.0
        self.use_scv = use_scv
        self.initialized = False
        self.current_warp = np.empty((0,0))

    cpdef initialize(self, double[:,:] img, double[:,:] region_corners):
        cpdef double ctx, cty, wwidth, hheight
        self.initialized = False
        
        ctx = np.mean(region_corners, axis=1)[0]
        cty = np.mean(region_corners, axis=1)[1]
        wwidth = (region_corners[0,1]+region_corners[0,2]-region_corners[0,3]-region_corners[0,0]) / 2.0
        hheight = (region_corners[1,3]+region_corners[1,2]-region_corners[1,0]-region_corners[1,1]) / 2.0
        self.est = np.array([[ctx, cty, wwidth/self.sz_T[0,0], 0.0, hheight/wwidth, 0.0]])
        self.est = affparam2mat(self.est)
        #self.est = square_to_corners_warp(np.asarray(region_corners), 4)
        self.param = np.array(np.tile(affparam2geom(self.est),[self.n_samples,1]))
        self.template = np.asarray(warpimg(img, self.est, self.sz_T)) #sz: 1*2 w,h !!!Wrong
        self.template = np.asarray(whiten(self.template))
        self.template /= np.linalg.norm(self.template)        

        self.initialized = True

    cpdef initialize_with_rectangle(self, double[:,:] img, ul, lr):
        cpdef double[:,:] region_corners = \
            np.array([[ul[0], ul[1]],
                      [lr[0], ul[1]],
                      [lr[0], lr[1]],
                      [ul[0], lr[1]]], dtype=np.float64).T
        self.initialize(img, region_corners)

    cpdef update(self, double[:,:] img):
        if not self.initialized: return
        cdef double[:,:] samples
        cdef double[:] p
        cdef int n
        a1 = time.time()
        samples, self.param = estwarp_condens(img, self.param, self.n_samples, self.sz_T, self.affsig)
        a2 = time.time()
        wlist = weight_eval(samples, self.template, self.sz_T, self.alpha)
        a3 = time.time()
        #wlist = est_weight_warp(img, self.template, self.param, self.n_samples, self.sz_T, self.affsig, self.alpha)
        (q,indq) = des_sort(wlist)
        id_max = indq[0] #change here
        temp = np.zeros((1,6))
        temp[0,:] = self.param[id_max,:]
        self.est = affparam2mat(temp)
        p = np.zeros(self.n_samples,np.float64) #observation likelihood initialization
        n = 0        
        while (n<self.n_samples):
            if q[indq[n]] < 0:
                print('Prob should be positive')
                #sys.exit()
            p[indq[n]] = q[n]
            n += 1
        self.param = resample2(np.array(self.param), np.array(p))
        a4 = time.time()

        if self.use_scv:
            # change the sample method
            self.intensity_map = scv_intensity_map(sampled_img, self.template)

    cpdef set_intensity_map(self, double[:] intensity_map):
        self.intensity_map = intensity_map

    cpdef double[:] get_intensity_map(self):
        return self.intensity_map

    cpdef is_initialized(self):
        return self.initialized

    cpdef set_warp(self, double[:,:] warp, bint reset_intensity=True):
        self.current_warp = warp
        if reset_intensity: self.intensity_map = None

    cpdef double[:,:] get_warp(self):
        return np.asmatrix(self.current_warp)

    cpdef double[:,:] get_tmplt(self):
        return np.asarray(self.template.reshape(self.resx,self.resy))

    cpdef set_region(self, double[:,:] corners, bint reset_intensity=True):
        ctx = np.mean(corners, axis=1)[0]
        cty = np.mean(corners, axis=1)[1]
        wwidth = (corners[0,1]+corners[0,2]-corners[0,3]-corners[0,0]) / 2.0
        hheight = (corners[1,3]+corners[1,2]-corners[1,0]-corners[1,1]) / 2.0
        # dx, dy, sc, th, sr, phi
        self.est = np.array([[ctx, cty, wwidth/self.sz_T[0,0], 0.0, hheight/wwidth, 0.0]])
        self.est = affparam2mat(self.est)

        if reset_intensity: self.intensity_map = None

    cpdef get_region(self):
        return drawbox(self.est, self.sz_T)

