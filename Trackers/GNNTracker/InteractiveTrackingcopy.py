"""
Author: Travis Dick (travis.barry.dick@gmail.com)
"""
import os
import cv
import cv2
import numpy as np

from CascadeTracker import *
from Homography import *
from ImageUtils import *
from NNTracker import *

class InteractiveTrackingApp:
    def __init__(self, tracker, size, filename, trackername, name="vis"):
        """ An interactive window for initializing and visualizing tracker state.

        The on_frame method should be called for each new frame. Typically real
        applications subclass InteractiveTrackingApp and build in some application
        loop that captures frames and calls on_frame.
        
        Parameters:
        -----------
        tracker : TrackerBase
          Any class implementing the interface of TrackerBase. 

        name : string
          The name of the window. Due to some silliness in OpenCV this must
          be unique (in the set of all OpenCV window names).

        See Also:
        ---------
        StandaloneTrackingApp
        RosInteractiveTrackingApp
        """

        self.tracker = tracker
        self.name = name
        self.m_start = None
        self.m_end = None
        self.gray_img = None
        self.paused = False
        self.img = None
	self.times = 1
	if not os.path.exists(filename):
		os.mkdir(filename)
	self.fname = open('./'+filename+'/'+trackername+'.txt','w')
        self.fname.write('%-8s%-8s%-8s%-8s%-8s%-8s%-8s%-8s%-8s\n'%('frame','ulx','uly','urx','ury','lrx','lry','llx','lly'))
	cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self.mouse_handler)
	#self.writer = cv2.VideoWriter('alpha.avi',cv.CV_FOURCC('D','I','V','3'),10,size)

    def display(self, img):
        annotated_img = img.copy()
        if self.tracker.is_initialized():
            draw_region(annotated_img, self.tracker.get_region(), (0,255,0), 2)
	    corners = self.tracker.get_region()
            self.fname.write('%-15s%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f\n'%('frame'+('%05d'%(self.times))+'.jpg',corners[0,0],corners[1,0],corners[0,1],corners[1,1],corners[0,2],corners[1,2],corners[0,3],corners[1,3]))
	if self.m_start != None and self.m_end != None:
            ul = (min(self.m_start[0],self.m_end[0]), min(self.m_start[1],self.m_end[1]))
            lr = (max(self.m_start[0],self.m_end[0]), max(self.m_start[1],self.m_end[1]))            
            corners = np.array([ ul, [lr[0],ul[1]], lr, [ul[0],lr[1]]]).T
	    draw_region(annotated_img, corners, (255,0,0), 1)
	    self.fname.write('%-15s%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f\n'%('frame'+('%05d'%(self.times))+'.jpg',corners[0,0],corners[1,0],corners[0,1],corners[1,1],corners[0,2],corners[1,2],corners[0,3],corners[1,3]))
        cv2.imshow(self.name, annotated_img)
	
	if self.times == 1: cv.WaitKey(6000)
	#self.writer.write(annotated_img)
	cv.SaveImage('/home/administrator/Desktop/result_video/result_faceocc2/'+str(self.times)+'.jpg',cv.fromarray(annotated_img))

    def mouse_handler(self, evt,x,y,arg,extra):
        if self.gray_img == None: return 
        if evt == cv2.EVENT_LBUTTONDOWN and self.m_start == None:
            self.m_start = (x,y)
            self.m_end = (x,y)
            self.paused = True
        elif evt == cv2.EVENT_MOUSEMOVE and self.m_start != None:
            self.m_end = (x,y)
        elif evt == cv2.EVENT_LBUTTONUP:
            self.m_end = (x,y)
            ul = (min(self.m_start[0],self.m_end[0]), min(self.m_start[1],self.m_end[1]))
            lr = (max(self.m_start[0],self.m_end[0]), max(self.m_start[1],self.m_end[1]))
            self.tracker.initialize_with_rectangle(self.gray_img, ul, lr)
            self.m_start, self.m_end = None, None
            self.paused = False
            self.inited = True
	#cv.WaitKey(1000)

    def on_frame(self, img):
        if not self.paused:
            self.img = img
            self.gray_img = cv2.GaussianBlur(to_grayscale(img), (5,5), 3)
            #self.gray_img = to_grayscale(img)
            self.tracker.update(self.gray_img)
	#else:
	#    cv.WaitKey(10)
        if self.img != None: self.display(self.img)
        key = cv.WaitKey(7)
        if key == ord(' '): self.paused = not self.paused
        elif key > 0: return False
	self.times = self.times + 1
        return True

    def cleanup(self):
        #cv2.destroyWindow(self.name)
	self.fname.close()
	#self.writer.release
