"""
Implementation of the ESM tracking algorithm.

S. Benhimane and E. Malis, "Real-time image-based tracking of planes
using efficient second-order minimization," Intelligent Robots and Systems, 2004.
(IROS 2004). Proceedings. 2004 IEEE/RSJ International Conference on, vol. 1, 
pp. 943-948 vol. 1, 2004.

Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import numpy as np
from scipy.linalg import expm

from Homography import *
from ImageUtils import *
from SCVUtils import *
from SL3HomParam import make_hom_sl3
from TrackerBase import *

class ESMTracker(TrackerBase):
    
    def __init__(self, max_iters, threshold=0.01, res=(20,20), use_scv=False):
        
        self.max_iters = max_iters
        self.threshold = threshold
        self.res = res
        self.pts = res_to_pts(self.res)
        self.use_scv = use_scv
        self.initialized = False

    def set_region(self, corners):
        self.proposal = square_to_corners_warp(corners)

    def initialize(self, img, region):
        self.set_region(region)
        self.template = sample_region(img, self.pts, self.get_warp())
        self.Je = _estimate_jacobian(img, self.pts, self.proposal)
        self.intensity_map = None
        self.initialized = True

    def update(self, img):
        if not self.initialized: return
        for i in xrange(self.max_iters):
            sampled_img = sample_region(img, self.pts, self.get_warp())
            if self.use_scv and self.intensity_map != None: sampled_img = scv_expectation(sampled_img, self.intensity_map)
            error = np.asmatrix(self.template - sampled_img).reshape(-1,1)
            Jpc = _estimate_jacobian(img, self.pts, self.proposal)
            J = (Jpc + self.Je) / 2.0
            update = np.asarray(np.linalg.lstsq(J, error)[0]).squeeze()
            self.proposal = self.proposal * make_hom_sl3(update)
            if np.sum(np.abs(update)) < self.threshold: break
        self.intensity_map = scv_intensity_map(sample_region(img, self.pts, self.get_warp()),self.template)

    def is_initialized(self):
        return self.initialized

    def get_warp(self):
        return self.proposal

    def get_region(self):
        return apply_to_pts(self.get_warp(), np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T)

def _estimate_jacobian(img, pts, initial_warp, eps=1e-10):
    n_pts = pts.shape[1]
    def f(p):
        W = initial_warp * make_hom_sl3(p)
        return sample_region(img, pts, W)
    jacobian = np.empty((n_pts,8))
    for i in xrange(0,8):
        o = np.zeros(8)
        o[i] = eps
        jacobian[:,i] = (f(o) - f(-o)) / (2*eps)
    return np.asmatrix(jacobian)
