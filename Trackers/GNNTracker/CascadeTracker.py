"""
Author: Travis Dick (travis.barry.dick@gmail.com)
"""

from TrackerBase import *

class CascadeTracker(TrackerBase):

    def __init__(self, trackers):
        """ Allows multiple trackers to be combined in series.

        Given a sequence of tracking algorithms, the CascadeTracker
        will use each tracker to compute an incremental update to
        the region proposed by the previous tracker. This allows us
        to combine different algorithms in a "coarse-to-fine" tracking
        scheme, where first we roughly take care of large motions and
        then utilize some very precise trackers for the final alignment.

        Parameters:
        -----------
        trackers : [TrackerBase]
          trackers is a list of objects each implementing the TrackerBase
          interface.       

        See Also:
        ---------
        TrackerBase
        MultiProposalTracker
        ParallelTracker
        """
        self.trackers = trackers
        self.initialized = False
        
    def set_region(self, region):
        for t in self.trackers: t.set_region(region)
        self.region = region
    
    def initialize(self, img, region):
        for t in self.trackers: t.initialize(img, region)
        self.region = region
        self.initialized = True
        
    def update(self, img):
        if not self.initialized: return
        region = None
        for t in self.trackers:
            if region != None: t.set_region(region)
            t.update(img)
            region = t.get_region()
        self.region = region
    
    def is_initialized(self):
        return self.initialized
    
    def get_region(self):
        return self.region
