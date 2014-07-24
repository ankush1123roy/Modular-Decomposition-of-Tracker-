"""
A small standalone application for tracker demonstration. Depends
on OpenCV VideoCapture to grab frames from the camera.

Author: Travis Dick (travis.barry.dick@gmail.com)
"""
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
    
    def __init__(self, vc, tracker, filename, tracker_name, path,initparam,nframe,name = 'vis'):
        InteractiveTrackingApp.__init__(self, tracker, filename, tracker_name,name)
        self.vc = vc
	self.path = path
	self.initparam = initparam     
	self.nframe = nframe

    def run(self):
        # reading imgs
        #path = '/home/xzhang6/Documents/cvpr_dataset/Singer1/img/'
	#initparam = [[51,53],[138,53],[138,343],[51,343]]
        i = 1
        while i <= self.nframe:
            img = cv2.imread(self.path+'%05d.jpg'%i)
#	    import pdb;pdb.set_trace()
            #(succ, img) = self.vc.read()
            #if not succ: break
            if img == None: 
		print('error loading image')
		break
	    if not self.on_frame(img,i,self.initparam): break
            i += 1
	self.cleanup()

if __name__ == '__main__':
    coarse_tracker = NNTracker(100, 2, res=(40,40), use_scv=True)
    fine_tracker = ESMTracker(5, res=(40,40), use_scv=True)
    tracker = CascadeTracker([coarse_tracker, fine_tracker])
    #app = StandaloneTrackingApp(cv2.VideoCapture(0), coarse_tracker)
    filename = '/home/ankush/OriginalNN/Test/NNTracker/src/results/bookII/10000_1/'
    path = '/home/ankush/GNNTracker/Videos/nl_bookII_s3/frame'
    initparam = [[305.00,289.00],[400.00,284.00],[406.00,437.00],[312.00,440.00]]
#  1.0371429e+002  1.7800000e+002  4.7728571e+002  3.9228571e+002
#  1.2942857e+002  5.9014286e+002  5.3871429e+002  7.5857143e+001
  #1.6442857e+002  1.5800000e+002  3.5228571e+002  3.5800000e+002
  #4.3714286e+001  2.8300000e+002  2.8728571e+002  5.5857143e+001
    tracker_name = 'nl_bookII_s3_10000'
    nframe = 2045 
    
    app = StandaloneTrackingApp(None, coarse_tracker,filename,tracker_name,path,initparam,nframe)
    app.run()
    app.cleanup()
