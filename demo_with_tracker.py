#imports de demo
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from posix import RTLD_GLOBAL

import _init_paths

import os
import cv2
import json
import sys
import numpy as np

from opts import opts
from detectors.detector_factory import detector_factory

##imports de tracker
from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from tracker.mailer import Mailer
from tracker import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest
import torch
import pandas as pd
#from general import check_img_size, non_max_suppression, xyxy2xywh #(LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
#from torch_utils import select_device



image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
detector = None

def detect(opt,img):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    if detector is None:
        Detector = detector_factory[opt.task]
        detector = Detector(opt)

    ret = detector.run(img)
    time_str = ''
    results = ret["results"][1]
    for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
    print(time_str)
    return(results)

def track(opt):
    imgsz=opt.imgsz
    conf_thres=opt.conf_thres
    iou_thres=opt.iou_thres
    max_det=opt.max_det
    is_gpu=opt.is_gpu
    confidence =opt.confidence
    classes=None
    agnostic_nms=False
    half=False,  # use FP16 half-precision inference
    device = torch.device('cuda:0' if is_gpu else 'cpu')#select_device(is_gpu) #change by '' for CUDA devices
    #stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    #imgsz = check_img_size(imgsz, s=stride)  # check image size


    print("[INFO] Starting the video..")
    base_name = opt.input.replace("/content/","").replace(".mp4","")
    vs = cv2.VideoCapture(opt.input)

    # initialize the video writer (we'll instantiate later if need be)
    writer = None

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=2, maxDistance=50)
    trackers = []
    trackableObjects = {}

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0
    counts = 0
    x = []
    empty=[]
    empty1=[]
    list_of_point_dicts = []


    # start the frames per second throughput estimator
    fps = FPS().start()

    if config.Thread:
        vs = thread.ThreadingClass(config.url)

    # loop over frames from the video stream
    while True:
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        frame = vs.read()
        flag, rgb = frame
        

        if not flag:
            break
        
        print(rgb.shape)

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = rgb.shape[:2]

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if opt.output is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(opt.output, fourcc, 30,
                (W, H), True)

        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % opt.skip_frames == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []
            # Llama al detector de circulos

            preds = detect(opt,rgb)


            # loop over the detections
            for pred in reversed(preds):
              center_x = pred[0]
              center_y = pred[1]
              radio = pred[2]
              conf = pred[3]
              classl = pred[4]
              startX, startY = center_x-radio,center_y-radio
              endX, endY = center_x+radio,center_y+radio
              if conf > confidence:
                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                
                tracker_radio = (tracker,radio)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker_radio)
                rects.append((startX, startY, endX, endY, radio))
                      

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker_radio in trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker, radio = tracker_radio
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY, radio))

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects, radios = ct.update(rects)

        # loop over the tracked objects
        name = base_name + "_" + str(totalFrames) + ".png"
        for (objectID, centroid) in objects.items():
            radio = radios[objectID]
            if status == "Detecting": detecting = True
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid, radio, name, detecting)        
            # otherwise, there is a trackable object so we have to add the observation
            else:
                to.add_observation(centroid, radio, name, detecting)


            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(rgb, text, (centroid[0], centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(RTLD_GLOBAL, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
            cv2.circle(rgb, (centroid[0], centroid[1]), int(radio), (0, 255, 0), 3)
            

                
        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(rgb)
        
        if status == "Detecting":
            output_name = "output/" + name
            print(output_name)
            cv2.imwrite(output_name,rgb)

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    return trackableObjects

def post_processing(dict_to):
    output_list = []
    for id in dict_to:
        to = dict_to[id]
        observation_list = to.observations
        obs_dfs = []
        for obs_dict in observation_list:
            obs_dfs.append(pd.from_dict(obs_dict))
        to_df = pd.concat(obs_dfs)
        to_df["track_id"] = id
        to_df["label"] = "baya"
        output_list.append(to_df)
    output_df = pd.concat(output_list)
    return output_df

if __name__ == '__main__':
  #load options from argparse
  opt = opts().init()
  track_obj_dict = track(opt)
  output_df = post_processing(track_obj_dict)
  output_df.to_csv("output.csv")
