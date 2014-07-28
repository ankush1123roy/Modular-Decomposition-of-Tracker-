from nntracker.utility import *
from nntracker.trackers.NNTracker import *

def point_to_region(p, s):
    x, y = p
    ul = (x-s, y-s)
    lr = (x+s, y+s)
    return rectangle_to_region(ul, lr)

def evaluate_region(img, region):
    tracker = NNTracker(1, 1000, 50, 50, 0.06, 0.04, False)
    tracker.initialize(img, region)
    return tracker.get_warp_index().mean_pixel_variance()



