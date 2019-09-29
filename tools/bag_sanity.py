#!/usr/bin/python

from __future__ import print_function

import argparse
import rosbag
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
import sys, os, time


# arguments
parser = argparse.ArgumentParser()
parser.add_argument("bag", help="ROS bag file to extract")
parser.add_argument("--topic", default="/prophesee/camera/cd_events_buffer", help="Event topic")
args = parser.parse_args()

bag = rosbag.Bag(args.bag, 'r')
topics = bag.get_type_and_topic_info()[1].keys()
print ("Available topics:")
for topic in topics:
    print ("\t", topic)

if (args.topic not in topics):
    print ("Specified topic in not present in the bag file!")
    sys.exit(0)

last_ts = -1.0
last_message_id = -1
last_event_id = -1
ts_offset = -1.0
for i, (topic, msg, t) in enumerate(bag.read_messages()):
    if (topic == args.topic) and False:
        if (len(msg.events) == 0):
            print ("Empty packet:", i)
            continue

        ts = msg.events[0].ts.to_sec()
        if (last_ts < 0 and ts_offset < 0):
            last_ts = msg.events[-1].ts.to_sec()
            last_message_id = i
            ts_offset = ts
            continue

        if (abs(ts - last_ts) > 0.01):
            print ("Possible lost packets!", last_ts - ts_offset, ts - ts_offset, ts - last_ts,
                                        "(", last_message_id, "(", "->", i, ")")

        last_ts = msg.events[-1].ts.to_sec()
        last_message_id = i
        continue

    if topic == args.topic:
        if (len(msg.events) == 0):
            print ("Empty packet:", i)
            continue

        for j, e in enumerate(msg.events):
            ts = e.ts.to_sec()
            if (last_ts < 0 and ts_offset < 0):
                last_ts = ts
                last_message_id = i
                last_event_id = j
                ts_offset = ts
                continue
            if (ts < last_ts and False):
                print ("Events in message:", len(msg.events))
                print ("Events are unordered!", last_ts - ts_offset, ts - ts_offset,
                                                "(", last_message_id, "(", last_event_id, ")", "->", i, "(", j, ")", ")")
            if (abs(ts - last_ts) > 0.01):
                print ("Events in message:", len(msg.events))
                print ("Possible lost packets!", last_ts - ts_offset, ts - ts_offset, ts - last_ts,
                                                "(", last_message_id, "(", last_event_id, ")", "->", i, "(", j, ")", ")")
            last_ts = ts
            last_message_id = i
            last_event_id = j
