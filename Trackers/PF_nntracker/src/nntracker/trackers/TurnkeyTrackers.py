"""
This module contains functions for constructing trackers with various
parameter settings that seem to work well in typical situations.

Author: Travis Dick (travis.barry.dick@gmail.com)
"""

from BMICTracker import BMICTracker
from ESMTracker import ESMTracker
from NNTracker import NNTracker
from PFTracker import PFTracker

from ThreadedCascadeTracker import ThreadedCascadeTracker
from CascadeTracker import CascadeTracker

def make_nn_bmic(use_scv=False, res=(40,40), nn_iters=10, nn_samples=1000, bmic_iters=5):
    t1 = NNTracker(nn_iters, nn_samples, res[0], res[1], 0.06, 0.04, use_scv)
    t2 = NNTracker(nn_iters, nn_samples, res[0], res[1], 0.03, 0.02, use_scv)
    t3 = NNTracker(nn_iters, nn_samples, res[0], res[1], 0.015, 0.01, use_scv)
    t4 = BMICTracker(bmic_iters, 0.001, res[0], res[1], use_scv)
    return CascadeTracker([t1, t2, t3, t4], use_scv=use_scv)

def make_bmic(use_scv=False, res=(40,40), iters=30):
    return BMICTracker(iters, 0.001, res[0], res[1], use_scv)

def make_esm(use_scv=False, res=(40,40), iters=20):
    return ESMTracker(iters, 0.001, res[0], res[1], use_scv)

def make_pf(use_scv=False, res=(40,40), n_samples=200):
    return PFTracker(n_samples, res[0], res[1], use_scv)
