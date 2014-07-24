from BakerMatthewsICTracker import *
from CascadeTracker import *
from ESMTracker import *
from Homography import *
from InteractiveTracking import *
from MultiProposalTracker import *
from NNTracker import *
from ParallelTracker import *

class StandaloneTrackingApp(InteractiveTrackingApp):
    """ A demo program that uses OpenCV to grab frames. """

    def __init__(self, vc, tracker, filename, trackername, name="vis"):
        self.vc = vc
	self.times = 0
	(succ,img) = self.vc.read()
	height,width,temp = img.shape
        InteractiveTrackingApp.__init__(self, tracker, (height,width), filename, trackername, name)

    def run(self):
	#(succ, img) = self.vc.read()
	#(succ, img) = self.vc.read()
	#(succ, img) = self.vc.read()
        while True:
	    self.times = self.times + 1
            (succ, img) = self.vc.read()
            if not succ: break
            if not self.on_frame(img,self.times): break
		
        self.cleanup()

if __name__ == '__main__':
    coarse_tracker = NNTracker(10000, 2, res=(40,40), use_scv=True)
    #fine_tracker = ESMTracker(5, res=(40,40), use_scv=True)
    #single_tracker = BakerMatthewsICTracker(50, res=(40,40), use_scv=True)
    #tracker = CascadeTracker([coarse_tracker, fine_tracker])
    filename = 'nl_cereal_s1'
    trackername = {0:'bmic',1:'esm',2:'nnbmic'}
    tracker0 = BakerMatthewsICTracker(50, res=(40,40), use_scv=True)
    tracker1 = ESMTracker(5, res=(40,40), use_scv=True)
    tracker2 = CascadeTracker([coarse_tracker,tracker0]) 
    app = StandaloneTrackingApp(cv2.VideoCapture('./'+filename+'.avi'), tracker1, filename, trackername[0])
    app.run()
    app.cleanup()

