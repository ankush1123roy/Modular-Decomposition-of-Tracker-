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
		path = '/home/ankush/GNNTracker/Videos/nl_cereal_s3/frame'
		initparam = [[38.00,323.00],[202.00,308.00],[216.00,517.00],[54.00,540.00]]
		i = 1
		while i <= 1200:
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
	coarse_tracker = NNTracker(5000, 2, res=(40,40), use_scv=True)
	fine_tracker = BakerMatthewsICTracker(40,res=(40,40),use_scv=True)
	# fine_tracker = ESMTracker(5, res=(40,40), use_scv=True)
	tracker = CascadeTracker([coarse_tracker, fine_tracker])
	filename = '/home/ankush/GNNTracker/Results/GNN/Cereal_s3_5000_2/'     # Directory where the resultant images would be stored 
	tracker_name = 'nl_cereal_s3_2'
	app = StandaloneTrackingApp(None, coarse_tracker,filename,tracker_name)
	app.run()
	app.cleanup()
