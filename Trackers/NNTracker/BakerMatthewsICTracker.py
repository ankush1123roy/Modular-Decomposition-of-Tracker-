"""
Implementation of the Baker-Matthews Inverse Compositional Tracking Algorithm.

S. Baker and I. Matthews, "Equivalence and efficiency of image alignment algorithms", 
Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE 
Computer Society Conference on, vol. 1, pp. I-1090-I-1097 vol. 1, 2001.

Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import numpy as np
from scipy.linalg import expm

from Homography import *
from ImageUtils import *
from SCVUtils import *
from SL3HomParam import make_hom_sl3
from TrackerBase import *

class BakerMatthewsICTracker(TrackerBase):

    def __init__(self, max_iters, threshold=0.01, res=(20,20), use_scv=False):
        """ An implementation of the inverse composititionl tracker from Baker and Matthews.

        Parameters:
        -----------
        max_iters : integer
          The maximum number of iterations per frame

        threshold : real
          If the decrease in error is smaller than the threshold then the per-frame
          update terminates.

        res : (integer, integer)
          The desired resolution of the template image. Higher values allow for more
          precise tracking but increase running time.
        
        See Also:
        ---------
        TrackerBase
        NNTracker
        """
        
        self.max_iters = max_iters
        self.res = res
        self.pts = res_to_pts(self.res)
        self.n_pts = np.prod(res)
        self.initialized = False
        self.use_scv = use_scv

    def set_region(self, corners):
        self.proposal = square_to_corners_warp(corners)
        
    def initialize(self, img, region):
        self.set_region(region)
        self.template = sample_region(img, self.pts, self.get_warp())
        self.VT_dW_dp = _estimate_jacobian(img, self.pts, self.proposal)  
        H = self.VT_dW_dp.T * self.VT_dW_dp
        self.H_inv = H.I      
        self.initialized = True

        # Image Gradient:
        #nabla_T = image_gradient(img, self.pts, self.get_warp())
        # Steepest Descent Images:
        #self.VT_dW_dp = np.empty((self.n_pts, 8))
        #for i in xrange(self.n_pts):
        #    self.VT_dW_dp[i,:] = np.asmatrix(nabla_T[i,:]) * _make_hom_jacobian(self.pts[:,i])
        #self.VT_dW_dp = np.asmatrix(self.VT_dW_dp)

        # Hessian:
        
        # H = np.zeros((8,8))
        # for i in xrange(self.n_pts):
        #     H += np.asmatrix(self.VT_dW_dp[i,:].T) * self.VT_dW_dp[i,:]
        # self.H_inv = np.asmatrix(H).I

    def update(self, img):
        if not self.is_initialized(): return None

        for i in xrange(self.max_iters):
            IWxp = sample_region(img, self.pts, self.get_warp())
            if self.use_scv: IWxp = scv_expectation(IWxp, self.template)
            error_img = np.asmatrix(IWxp - self.template)
            update = np.asmatrix(self.VT_dW_dp.T)*error_img.reshape((-1,1))
            update = self.H_inv * np.asmatrix(update).reshape((-1,1))
            update = np.asarray(update).squeeze()
            self.proposal = self.proposal * make_hom_sl3(update).I

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

import numpy as np
from scipy.linalg import expm



# def _make_hom(p):
#     return np.matrix([[1 + p[0],   p[1]  , p[2]],
#                       [  p[3]  , 1 + p[4], p[5]],
#                       [  p[6]  ,   p[7]  ,   1]], dtype=np.float64)

# def _make_hom_jacobian(pt, p=np.zeros(8)):
#     x = pt[0]
#     y = pt[1]
#     jacobian = np.zeros((2,8))
#     d = (p[6]*x + p[7]*y + 1)
#     d2 = d*d
#     jacobian[0,0] = x / d
#     jacobian[0,1] = y / d
#     jacobian[0,2] = 1 / d
#     jacobian[1,3] = x / d
#     jacobian[1,4] = y / d
#     jacobian[1,5] = 1 / d
#     jacobian[0,6] = -x * (p[0]*x + x + p[1]*y + p[2]) / d2
#     jacobian[0,7] = -y * (p[0]*x + x + p[1]*y + p[2]) / d2
#     jacobian[1,6] = -x * (p[3]*x + p[4]*y + y + p[5]) / d2
#     jacobian[1,7] = -y * (p[3]*x + p[4]*y + y + p[5]) / d2
#     return np.asmatrix(jacobian)
