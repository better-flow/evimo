#!/usr/bin/python

import argparse
import rosbag
import rospy
import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import os

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("bag", help="ROS bag file to extract")
parser.add_argument("--event_topic", default="/prophesee/camera/cd_events_buffer", help="Event topic")
parser.add_argument("--imu_topic", default="/unknown/imu", help="IMU topic")
args = parser.parse_args()

# Use a CvBridge to convert ROS images to OpenCV images so they can be saved.
bridge = CvBridge()

def get_filename(i):
    return "images/frame_" + str(i).zfill(8)  + ".png"

def get_depth_filename(i):
    return "depthmaps/frame_" + str(i).zfill(8)  + ".exr"

def timestamp_str(ts):
    return str(ts.secs) + "." + str(ts.nsecs).zfill(9)


image_index = 0
depthmap_index = 0
event_sum = 0
imu_msg_sum = 0
groundtruth_msg_sum = 0
calib_written = False

events_file = open('events.txt', 'w')
imu_file = open('imu.txt', 'w')

with rosbag.Bag(args.bag, 'r') as bag:
    # reset time?
    reset_time = rospy.Time()
    if True:
        first_msg = True
        for topic, msg, t in bag.read_messages():
            got_stamp = False
            if topic == args.event_topic:
                stamp = msg.events[0].ts
                got_stamp = True
            elif topic == args.imu_topic:
                stamp = msg.header.stamp
                got_stamp = True

            if got_stamp:
                if first_msg:
                    reset_time = stamp
                    first_msg = False
                else:
                    if stamp < reset_time:
                        reset_time = stamp
			break
    print "Reset time: " + timestamp_str(reset_time)

    for topic, msg, t in bag.read_messages():
        # events
        if topic == args.event_topic:
            for e in msg.events:
                events_file.write(timestamp_str(e.ts - reset_time) + " ")
                events_file.write(str(e.x) + " ")
                events_file.write(str(e.y) + " ")
                events_file.write(("1" if e.polarity else "0") + "\n")
                event_sum = event_sum + 1

        # IMU
        elif topic == args.imu_topic:
            imu_file.write(timestamp_str(msg.header.stamp - reset_time) + " ")
            imu_file.write(str(msg.linear_acceleration.x) + " ")
            imu_file.write(str(msg.linear_acceleration.y) + " ")
            imu_file.write(str(msg.linear_acceleration.z) + " ")
            imu_file.write(str(msg.angular_velocity.x) + " ")
            imu_file.write(str(msg.angular_velocity.y) + " ")
            imu_file.write(str(msg.angular_velocity.z) + "\n")
            imu_msg_sum = imu_msg_sum + 1


# statistics (remove missing groundtruth or IMU file if not available)
print "All data extracted!"
print "Events:       " + str(event_sum)
print "IMU:          " + str(imu_msg_sum)

# close all files
events_file.close()
imu_file.close()

# clean up
if imu_msg_sum == 0:
    os.remove("imu.txt")
    print "Removed IMU file since there were no messages."

