"""
FImplementation of the Nearest Neighbour Tracking Algorithm.
Author: Travis Dick (travis.barry.dick@gmail.com)
        Ankush Roy (ank ush2@ualberta.ca)
        Xi Zhang (xzhang6@ualberta.ca)
"""
import time
import numpy as np
import pyflann
from scipy import weave
from scipy.weave import converters
from numpy import matrix as MA
from Homography import *
from ImageUtils import *
from SCVUtils import *
from TrackerBase import *

import cv2
import pdb
import sys

from build_graph import *
from search_graph import *
from knnsearch import *

class NNTracker(TrackerBase):

    def __init__(self, n_samples, n_iterations=10, res=(40,40),
                 warp_generator=lambda:random_homography(0.04, 0.04),
                 use_scv=False):
        """ An implemetation of the Nearest Neighbour Tracker. 

        Parameters:
        -----------
        n_samples : integer
          The number of sample motions to generate. Higher values will improve tracking
          accuracy but increase running time.
        
        n_iterations : integer
          The number of times to update the tracker state per frame. Larger numbers
          may improve convergence but will increase running time.
        
        res : (integer, integer)
          The desired resolution of the template image. Higher values allow for more
          precise tracking but increase running time.

        warp_generator : () -> (3,3) numpy matrix.
          A function that randomly generates a homography. The distribution should
          roughly mimic the types of motions that you expect to observe in the 
          tracking sequence. random_homography seems to work well in most applications.
          
        See Also:
        ---------
        TrackerBase
        BakerMatthewsICTracker
        """
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.res = res
        self.resx = res[0]
        self.resy = res[1]
        self.warp_generator = warp_generator
        self.n_points = np.prod(res)
        self.initialized = False
        self.pts = res_to_pts(self.res)
        self.use_scv=use_scv
        self.gnn = True
 
    def set_region(self, corners):
        self.proposal = square_to_corners_warp(corners)

    def initialize(self, img, region):
        self.set_region(region)
        self.template = sample_and_normalize(img, self.pts, self.get_warp())
        #Jesse
	self.warp_index = _WarpIndex(self.n_samples, self.warp_generator, img, self.pts, self.get_warp(),self.res)

        self.intensity_map = None
        self.initialized = True

    def is_initialized(self):
        return self.initialized

    def update(self, img):
        if not self.is_initialized(): return None
        for i in xrange(self.n_iterations):
            sampled_img = sample_and_normalize(img, self.pts, warp=self.proposal)
            if self.use_scv and self.intensity_map != None: sampled_img = scv_expectation(sampled_img, self.intensity_map)
            # --sift-- #
            if self.gnn == True:
                self.proposal = self.proposal * self.warp_index.best_match(sampled_img)
                self.proposal /= self.proposal[2,2]
            else:
		temp_desc = self.pixel2sift(sampled_img)
		update = self.desc2warp_weighted3(temp_desc)
		self.proposal = self.proposal * update
                self.proposal /= self.proposal[2,2]
        if self.use_scv: self.intensity_map = scv_intensity_map(sample_region(img, self.pts, self.get_warp()), self.template)
        return self.proposal

    def get_warp(self):
        return self.proposal

    def get_region(self):
        return apply_to_pts(self.get_warp(), np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T)
    #-- sift --#
    def pixel2sift(self,patch):
        detector = cv2.FeatureDetector_create("SIFT")
        detector.setDouble('edgeThreshold',30)
	descriptor = cv2.DescriptorExtractor_create("SIFT")
        #sift = cv2.SIFT(edgeThreshold = 20)
	patch = (patch.reshape(self.resx,self.resy)).astype(np.uint8)
        skp = detector.detect(patch)
        skp, sd = descriptor.compute(patch, skp)
	return sd

    # --- For sift --- #
    def desc2warp_weighted(self,descs):
        warps = np.zeros((3,3), dtype=np.float64)
        temp_desc = np.empty((128,1),dtype=np.float32)
	if descs == None:
	    print('The number of descriptors is zero!')
	    return np.eye(3,dtype=np.float32)
	for i in range(descs.shape[0]):
            temp_desc[:,0] = descs[i,:]
            warp,dist = self.warp_index.best_match_sift(temp_desc.T)
            warps += warp
        warps /= descs.shape[0]
        return warps

    # --- For sift --- #
    def desc2warp_weighted2(self,descs):
        warps= np.zeros((3,3), dtype=np.float64)
        temp_desc = np.empty((128,1),dtype=np.float32)
        if descs == None:
            print('The number of descriptors is zero!')
            return np.eye(3,dtype=np.float32)
	warp_list = []
	dist_list = []
        for i in range(descs.shape[0]):
            temp_desc[:,0] = descs[i,:]
            warp,dist = self.warp_index.best_match_sift(temp_desc.T)
            #warps += warp
	    warp_list.append(warp)
	    dist_list.append(dist)
	thres = max(dist_list) * 0.5
	count = 0
	for i in range(len(dist_list)):
	    if dist_list[i] <= thres:
		warps += warp_list[i]
		count += 1
	if count == 0: return np.eye(3,dtype=np.float32)
        warps /= count
        return warps

    # --- For sift --- #
    def desc2warp_weighted3(self,descs):
        warps = np.zeros((3,3), dtype=np.float64)
        temp_desc = np.empty((128,1),dtype=np.float32)
        if descs == None:
            print('The number of descriptors is zero!')
            return np.eye(3,dtype=np.float32)
        warp_list = []
        dist_list = [] 
	print('Testing')
        for i in range(descs.shape[0]):
            temp_desc[:,0] = descs[i,:]
            warp,dist,index = self.warp_index.best_match_sift(temp_desc.T)
#            print(index)
	    #warps += warp
            warp_list.append(warp)
            dist_list.append(dist)
	sum_dist = sum(dist_list)
        for i in range(len(dist_list)):
	    warps += warp_list[i] * dist_list[i] /sum_dist
        return warps

class _WarpIndex:
    """ Utility class for building and querying the set of reference images/warps. """
    def __init__(self, n_samples, warp_generator, img, pts, initial_warp,res):
	self.nodes = None
	self.resx = res[0]
	self.resy = res[1]
        self.gnn = True
        self.indx = []
	n_points = pts.shape[1]
        print "Sampling Warps..."
        self.warps = [np.asmatrix(np.eye(3))] + [warp_generator() for i in xrange(n_samples - 1)]
        print "Sampling Images..."
        self.images = np.empty((n_points, n_samples))
        for i,w in enumerate(self.warps):
            self.images[:,i] = sample_and_normalize(img, pts, initial_warp * w.I)
        print('Graph based Nearest Neighbour')
        print('------------------------------')
        if self.gnn == True:
            #self.flann = pyflann.FLANN()
	    #print(self.images.shape)
            #self.flann.build_index(self.images.T, algorithm='kdtree', trees=10)
	    self.images = MA(self.images,'f8')
	    self.nodes = build_graph(self.images.T,40)
           
        else:
            desc = self.list2array(self.pixel2sift(self.images))
            # --- Building Flann Index --- #
            self.flann = pyflann.FLANN()
        print "Done!"

    # --- For sift --- #
    def pixel2sift(self,images):
        detector = cv2.FeatureDetector_create("SIFT")
	detector.setDouble('edgeThreshold',30)
        descriptor = cv2.DescriptorExtractor_create("SIFT")
        #sift = cv2.SIFT(edgeThreshold = 20)
	# -- store descriptors in list --#
        desc = []
        for i in range(images.shape[1]):
            patch = (images[:,i].reshape(self.resx,self.resy)).astype(np.uint8)
	    skp = detector.detect(patch)
            skp, sd = descriptor.compute(patch, skp)
            desc.append(sd)
            self.indx.append(len(skp))
	return desc

    # --- For sift ---#
    def list2array(self,desc):
        nums = sum(self.indx)
        descs = np.empty((128,nums),dtype=np.float64)
        counts = 0
        for item in desc:
            if item == None:
		continue
	    for j in range(item.shape[0]):
                descs[:,counts] = item[j,:].T
                counts += 1
        return descs.astype(np.float32)

    # ---SIFT function --- #
    def best_match_sift(self, desc):
	  results, dists = self.flann.nn_index(desc)
	  index = int(results[0])
          index += 1
          count = 0
	  for item in self.indx:
	      if index <= item:
                  result = count
              else:
                  index -= item
                  count += 1
          return self.warps[count], dists[0], count

    def best_match(self, img):
        t1 = time.time()
        nn_id,b,c = search_graph(img,self.nodes,self.images.T,1)
        t2 = time.time()
        #print t2 - t1
        return self.warps[int(nn_id)]
