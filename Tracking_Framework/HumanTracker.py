__author__ = 'Tommy'

import cv2
import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import Tkinter as tk
import os
import sys
import math
from TrackerBase import *

class HumanTracker(TrackerBase):

    def __init__(self,img, no_of_frames, init_corners):
        self.rot_angle=np.pi/12
        self.scale_up_factor=1.01
        self.scale_down_factor=0.99
        self.scale_up_mat=np.array([[self.scale_up_factor, 0], [0, self.scale_up_factor]])
        self.scale_down_mat=np.array([[self.scale_down_factor, 0], [0, self.scale_down_factor]])

        self.scale_up=False
        self.scale_down=False

        self.paused=True
        self.exit=False
        self.reset=False
        self.no_of_frames=no_of_frames
        self.mouse_pos=(0,0)
        self.img=img
        self.width=self.img.shape[0]
        self.height=self.img.shape[1]
        self.init_corners=init_corners.copy()
        self.init_center=np.mean(self.init_corners, axis=1)
        self.no_of_pts=init_corners.shape[1]
        self.curr_corners=init_corners.copy()
        self.curr_corners_3d=np.vstack((self.curr_corners,np.zeros([1,self.no_of_pts])))
        self.curr_center=self.init_center.copy()
        self.curr_center_3d=np.mean(self.curr_corners_3d, axis=1)
        self.prev_corners=init_corners.copy()
        self.prev_corners_3d=np.vstack((self.prev_corners,np.zeros([1,self.no_of_pts])))
        self.prev_center=self.init_center.copy()
        self.center=np.zeros([2, self.no_of_pts])
        self.center_3d=np.zeros([3, self.no_of_pts])
        #print "current_corners_3d=", self.curr_corners_3d

    def keyboardHandler(self):
        key=cv2.waitKey(1)
        #print "key=", key
        if key==27:
            manual_tracker.exit=True
        elif key==122:
            print "rotating around x axis clockwise by ",self.rot_angle
            self.rotateCorners(0, False)
        elif key==120:
            print "rotating around y axis clockwise by ",self.rot_angle
            self.rotateCorners(1, False)
        elif key==99:
            print "rotating around z axis clockwise by ",self.rot_angle
            self.rotateCorners(2, False)
        elif key==90:
            print "rotating around x axis anti-clockwise by ",self.rot_angle
            self.rotateCorners(0, True)
        elif key==88:
            print "rotating around y axis anti-clockwise by ",self.rot_angle
            self.rotateCorners(1, True)
        elif key==67:
            print "rotating around z axis anti-clockwise by ",self.rot_angle
            self.rotateCorners(2, True)
        elif key==82 or key==114:
            self.resetCorners()
        elif key==65:
            self.scaleCorners(True)
        elif key==97:
            self.scaleCorners(False)
        elif key==32:
            self.paused=not self.paused


    def mouseHandler(self, event, x, y, flags=None, userdata=None):
        #print "Left mouse button down at pos ", x, ", ", y
        #print "init_corners before=",  self.init_corners

        #print "init_corners after=",  self.init_corners
        #print "current_corners=",  self.curr_corners
        if event==cv2.EVENT_LBUTTONDOWN:
            self.scale_up=True
            #self.paused=False
            #self.scaleCorners(True)
        elif event==cv2.EVENT_LBUTTONUP:
            self.scale_up=False
            #self.paused=True
            #print "Left mouse button up at pos ", x, ", ", y
        elif event==cv2.EVENT_RBUTTONDOWN:
            self.scale_down=True
            #self.scaleCorners(False)
        elif event==cv2.EVENT_RBUTTONUP:
            self.scale_down=False
            #print "Right mouse button up at pos ", x, ", ", y
        elif event==cv2.EVENT_MBUTTONDOWN:
            self.paused=not self.paused
        elif event==cv2. EVENT_MOUSEMOVE:
            if not self.paused:
                self.mouse_pos=(int(x), int(y))
                if not np.array_equal(self.mouse_pos,self.curr_center):
                    #print "translating center to:",  self.mouse_pos
                    self.translateCorners()

    def updateFrame(self, img_id):
        if self.exit:
            sys.exit()
        if self.scale_up:
            self.scaleCorners(True)
        if self.scale_down:
            self.scaleCorners(False)

        self.keyboardHandler()
        self.img=cv2.imread(img_path+'/img%d.jpg'%img_id)
        actual_corners = [ground_truth[img_id-1, 0:2].tolist(),
        ground_truth[img_id-1, 2:4].tolist(),
        ground_truth[img_id-1, 4:6].tolist(),
        ground_truth[img_id-1, 6:8].tolist()]
        actual_corners=np.array(actual_corners).T
        actual_center=np.mean(actual_corners, axis=1)
        draw_region(self.img, actual_corners, (0, 255, 0), 2)
        cv2.circle(self.img, (int(actual_center[0]), int(actual_center[1])), 0, (0, 255, 0), 3)

        draw_region(self.img, self.curr_corners, (0, 0, 255), 2)
        cv2.circle(self.img, (int(self.curr_center[0]), int(self.curr_center[1])), 0, (0, 0, 255), 3)

    def updateState(self):
        self.curr_center=np.mean(self.curr_corners, axis=1)
        self.curr_corners_3d[0:2,:]=self.curr_corners
        self.curr_center_3d==np.mean(self.curr_corners_3d, axis=1)

        for i in xrange(self.no_of_pts):
            for j in xrange(2):
                self.center[j,i]=self.curr_center[j]
            for j in xrange(3):
                self.center_3d[j,i]=self.curr_center_3d[j]

    def resetCorners(self):
        self.curr_corners=self.init_corners.copy()
        self.updateState()

    def translateCorners(self):
        diff=self.mouse_pos-self.curr_center
        for i in xrange(self.no_of_pts):
            self.curr_corners[:,i]=self.curr_corners[:,i]+diff
        self.updateState()

    def scaleCorners(self, scale_up=False):
        self.curr_corners=self.curr_corners-self.center
        if scale_up:
            self.curr_corners=np.dot(self.scale_up_mat,self.curr_corners)
        else:
            self.curr_corners=np.dot(self.scale_down_mat,self.curr_corners)
        self.curr_corners=self.curr_corners+self.center
        self.updateState()

    def rotateCorners(self, axis, clockwise=False):
        if clockwise:
            angle_cos=math.cos(-self.rot_angle)
            angle_sin=math.sin(-self.rot_angle)
        else:
            angle_cos=math.cos(self.rot_angle)
            angle_sin=math.sin(self.rot_angle)

        if axis==0:
            rot_mat=np.array([[1, 0, 0], [0, angle_cos, -angle_sin], [0, angle_sin, angle_cos]])
        elif axis==1:
             rot_mat=np.array([[angle_cos, 0, angle_sin], [0, 1, 0], [-angle_sin, 0, angle_cos]])
        elif axis==2:
            rot_mat=np.array([[angle_cos, -angle_sin, 0], [angle_sin, angle_cos, 0], [0, 0, 1]])
        else:
            print "Invalid axis specified: ", axis
            return

        #self.prev_corners=self.curr_corners.copy()
        #self.prev_center=self.curr_center.copy()
        #self.prev_corners_3d=self.curr_corners_3d.copy()

        self.curr_center_3d=np.mean(self.curr_corners_3d, axis=1)
        self.curr_corners_3d=self.curr_corners_3d-self.center_3d
        self.curr_corners_3d=np.dot(rot_mat,self.curr_corners_3d)
        self.curr_corners_3d=self.curr_corners_3d+self.center_3d
        self.curr_corners=self.curr_corners_3d[0:2,:]

        self.updateState()

def readTrackingData(filename):
    if not os.path.isfile(filename):
        print "Tracking data file not found:\n ",filename
        sys.exit()

    data_file = open(filename, 'r')
    data_file.readline()
    lines = data_file.readlines()
    no_of_lines = len(lines)
    data_array = np.empty([no_of_lines, 8])
    line_id = 0
    for line in lines:
        #print(line)
        words = line.split()
        if (len(words) != 9):
            msg = "Invalid formatting on line %d" % line_id + " in file %s" % filename + ":\n%s" % line
            raise SyntaxError(msg)
        words = words[1:]
        coordinates = []
        for word in words:
            coordinates.append(float(word))
        data_array[line_id, :] = coordinates
        #print words
        line_id += 1
    data_file.close()
    return data_array

def draw_region(img, corners, color, thickness=1):
    #print "corners=", corners
    for i in xrange(4):
        p1 = (int(corners[0,i]), int(corners[1,i]))
        p2 = (int(corners[0,(i+1)%4]), int(corners[1,(i+1)%4]))
        #print "p1=", p1
        #print "p2=", p2
        cv2.line(img, p1, p2, color, thickness)

def getCenter(corners):
    center=np.mean(corners, axis=1)
    return center


if __name__ == '__main__':
    #init_frame = 0
    dataset_path = 'G:/UofA/Thesis/#Code/Datasets/Human'
    data_file='nl_bookII_s3'
    img_path=dataset_path+"/"+data_file
    ground_truth = readTrackingData(dataset_path + '/' + data_file + '.txt')
    no_of_frames = ground_truth.shape[0]
    init_img=cv2.imread(img_path+'/img1.jpg')
    width=init_img.shape[0]
    height=init_img.shape[1]
    init_corners = [ground_truth[0, 0:2].tolist(),
         ground_truth[0, 2:4].tolist(),
         ground_truth[0, 4:6].tolist(),
         ground_truth[0, 6:8].tolist()]
    init_corners=np.array(init_corners).T

    manual_tracker=HumanTracker(init_img,no_of_frames, init_corners)
    window_name="Manual Tracker CV"
    cv2.namedWindow(window_name)


    cv2.setMouseCallback(window_name, manual_tracker.mouseHandler)

    img_id=1
    while img_id<=no_of_frames:
        cv2.imshow(window_name,manual_tracker.img)
        if not manual_tracker.paused:
            img_id+=1
        if img_id>no_of_frames:
            img_id=1
        manual_tracker.updateFrame(img_id)












