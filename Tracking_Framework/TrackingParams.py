__author__ = 'Tommy'
import cv2
import numpy as np
from MultiProposalTracker import *
from NNTracker import *
from ParallelTracker import *
from BakerMatthewsICTracker import *
from CascadeTracker import *
from ESMTracker import *
from L1Tracker import *

class TrackingParams:
    def __init__(self, type):
        self.type = type
        self.params = []
        self.tracker=None

        self.update = lambda: None
        self.validate = lambda: True
        if type == 'nn':
            print 'Initializing NN tracker'
            self.params.append(Param('no_of_samples', 500,'int'))
            self.params.append(Param('no_of_iterations', 2,'int'))
            self.params.append(Param('resolution_x',40,'int'))
            self.params.append(Param('resolution_y', 40,'int'))
            self.update=lambda: NNTracker(self.params[0].val, self.params[1].val,
                                         res=(self.params[2].val, self.params[3].val),
                                         use_scv=False)
        elif type == 'esm':
            print 'Initializing ESM tracker'
            self.params.append(Param('max_iterations', 5,'int'))
            self.params.append(Param('threshold', 0.1,'float'))
            self.params.append(Param('resolution_x',40,'int'))
            self.params.append(Param('resolution_y', 40,'int'))
            self.update=lambda: ESMTracker(self.params[0].val, threshold=self.params[1].val,
                                         res=(self.params[2].val, self.params[3].val),
                                         use_scv=False)
        elif type == 'ict':
            print 'Initializing ICT tracker'
            self.params.append(Param('max_iterations', 10,'int'))
            self.params.append(Param('threshold', 0.1,'float'))
            self.params.append(Param('resolution_x',40,'int'))
            self.params.append(Param('resolution_y', 40,'int'))
            self.update=lambda: BakerMatthewsICTracker(self.params[0].val, threshold=self.params[1].val,
                                         res=(self.params[2].val, self.params[3].val),
                                         use_scv=False)
        elif type == 'l1':
            print 'Initializing L1 tracker'
            self.params.append(Param('no_of_samples', 10,'int'))
            self.params.append(Param('angle_threshold', 50,'float'))
            self.params.append(Param('resolution_x',12,'int'))
            self.params.append(Param('resolution_y', 15,'int'))
            self.params.append(Param('no_of_templates',10,'int'))
            self.params.append(Param('alpha', 50,'float'))
            self.update = lambda: L1Tracker(self.params[0].val, self.params[1].val,
                                            [[self.params[2].val, self.params[3].val]],
                                            self.params[4].val, self.params[5].val)
        else:
            self.init_success = False
            print "Invalid tracker:", type

    def printParamValue(self):
        #print self.type, "filter:"
        for i in xrange(len(self.params)):
            print self.params[i].name, "=", self.params[i].val


class Param:
    def __init__(self, name, val, type):
        self.name = name
        self.val = val
        self.type = type
