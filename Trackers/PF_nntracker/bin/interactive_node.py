#!/usr/bin/env python

import roslib; roslib.load_manifest("nntracker")
import rospy; rospy.init_node("interactive_node")

from sensor_msgs.msg import Image
from nntracker.msg import NNTrackerROI

from cv_bridge import CvBridge
import numpy as np

from nntracker.trackers.TurnkeyTrackers import *
from nntracker.InteractiveTracking import *
from nntracker.utility import *

class RosInteractiveTrackingApp(InteractiveTrackingApp):
    
    def __init__(self, trackers):
        InteractiveTrackingApp.__init__(self, trackers, "ROS Interactive Tracking", init_with_rectangle=False)
        self.image_sub = rospy.Subscriber("image", Image, self.image_callback)
        self.roi_pub = rospy.Publisher("roi_NNTracker", NNTrackerROI)
        self.bridge = CvBridge()

    def image_callback(self, img_msg):
        self.on_frame(np.array(self.bridge.imgmsg_to_cv(img_msg, "bgr8")))
        if self.trackers[0].is_initialized():
            message = NNTrackerROI()
            region = self.trackers[0].get_region()
            message.ulx, message.uly = region[:,0]
            message.urx, message.ury = region[:,1]
            message.lrx, message.lry = region[:,2]
            message.llx, message.lly = region[:,3]
            self.roi_pub.publish(message)

app = RosInteractiveTrackingApp([make_nn_bmic(use_scv=True)])

if __name__ == "__main__":
    rospy.spin()
    app.cleanup()


