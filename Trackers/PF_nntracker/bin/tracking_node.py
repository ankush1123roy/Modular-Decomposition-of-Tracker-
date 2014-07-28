#!/usr/bin/env python

import roslib
roslib.load_manifest('nntracker')

import threading

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import rospy

from nntracker.trackers.TurnkeyTrackers import *
from nntracker.msg import NNTrackerROI, NNTrackerCommand
from nntracker.utility import *

rospy.init_node("nntracker")

# ---------- Create the Tracker ---------- #
algorithm_name = rospy.get_param("~algorithm", "nn_bmic")
if algorithm_name == "nn_bmic":
    tracker = make_nn_GN(use_scv=True)
elif tracker_name == "esm":
    tracker = make_esm(use_scv=True)

# ---------- Set up the OpenCV Bridge ---------- #
cvbridge = CvBridge()

# ---------- Make a publisher to broadcast the ROI ---------- #
roi_pub = rospy.Publisher("roi", NNTrackerROI)

# ---------- Make a subscriber to grab images ---------- #

current_rgb_img = None
current_gray_img = None
def image_callback(data):
    global current_rgb_img, current_gray_img
    # Read image from message
    current_img = np.array(cvbridge.imgmsg_to_cv(data, "bgr8"))
    # Compute the gray image
    current_gray_img = cv2.GaussianBlur(np.asarray(to_grayscale(current_img)), (5,5), 3)
    # If the tracker is running, ask it to update and publish roi
    if tracker.is_initialized(): 
        # update
        tracker.update(current_gray_img)
        # publish roi
        message = NNTrackerROI()
        region = tracker.get_region()
        message.ulx, message.uly = region[:,0]
        message.urx, message.ury = region[:,1]
        message.lrx, message.lry = region[:,2]
        message.llx, message.lly = region[:,3]
        roi_pub.publish(message)
image_sub = rospy.Subscriber("image", Image, image_callback, queue_size = 1)

# ---------- Make a subscriber to listen for commands ---------- #

def command_callback(message):
    if message.command == "initialize":
        r = message.region
        region = np.array([(r.ulx, r.uly),
                           (r.urx, r.ury),
                           (r.lrx, r.lry),
                           (r.llx, r.lly)]).T
        tracker.initialize(current_gray_img, region)
command_sub = rospy.Subscriber("command", NNTrackerCommand, command_callback)

    
# ---------- Wait until ROS quits ----------

rospy.spin()
