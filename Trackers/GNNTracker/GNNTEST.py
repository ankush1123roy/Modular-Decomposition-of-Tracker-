"""
A small standalone application for tracker demonstration. Depends
on OpenCV VideoCapture to grab frames from the camera.

Author: Travis Dick (travis.barry.dick@gmail.com)
"""
import time
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
    
	def __init__(self, vc, tracker, filename, tracker_name,  name="vis"):
		InteractiveTrackingApp.__init__(self, tracker, filename, tracker_name, name)
		self.vc = vc
    
	def run(self):
        # reading imgs
		path = '/home/ankush/GNNTracker/Videos/nl_bookI_s3/frame'
		initparam = [[315,311],[451,306],[457,494],[326,502]]
		ST = time.time()
		i = 1
		while i <= 340:
			img = cv2.imread(path+'%05d.jpg'%i)
			if img == None: 
				print('error loading image')
				break
			if not self.on_frame(img,i,initparam): break
			i += 2
		SP = time.time()
		print  SP -ST
		self.cleanup()

if __name__ == '__main__':
	coarse_tracker = NNTracker(100, 2, res=(40,40), use_scv=True)
	fine_tracker = ESMTracker(5, res=(40,40), use_scv=True)
	tracker = CascadeTracker([coarse_tracker, fine_tracker])
	filename = '/home/ankush/GNNTracker/Results/GNN/bookI0.01std2000samples_5/'     # Directory where the resultant images would be stored 
	tracker_name = 'bookI0.01std1500samples_5'
	app = StandaloneTrackingApp(None, coarse_tracker,filename,tracker_name)
	app.run()
	app.cleanup()
