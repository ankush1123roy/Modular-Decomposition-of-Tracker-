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

    def __init__(self, vc, tracker, name="vis"):
        self.vc = vc
	(succ,img) = self.vc.read()
	height,width,temp = img.shape
        InteractiveTrackingApp.__init__(self, tracker, (height,width), name)

    def run(self):
	(succ, img) = self.vc.read()
	(succ, img) = self.vc.read()
	(succ, img) = self.vc.read()
        while True:
            (succ, img) = self.vc.read()
            if not succ: break
            if not self.on_frame(img): break
		
        self.cleanup()

if __name__ == '__main__':
    coarse_tracker = NNTracker(10000, 2, res=(40,40), use_scv=True)
    fine_tracker = ESMTracker(5, res=(40,40), use_scv=True)
    single_tracker = BakerMatthewsICTracker(50)
    tracker = CascadeTracker([coarse_tracker, fine_tracker])
    app = StandaloneTrackingApp(cv2.VideoCapture('./nl_bookII_s3.avi'), tracker)
    app.run()
    app.cleanup()

