#imports de demo
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import json
import sys
import numpy as np

from opts import opts
from detectors.detector_factory import detector_factory

##imports de tracker
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
from mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest




image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    while True:
        _, img = cam.read()
        cv2.imshow('input', img)
        ret = detector.run(img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]

    print(image_names)
    
    for image_name in image_names:
      ret = detector.run(image_name)
      time_str = ''
      results = ret["results"][1]
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
      return(results)

def run(opt):

    def parse_opt():
        parser = argparse.ArgumentParser(
                                        formatter_class=argparse.RawDescriptionHelpFormatter,
                                        description='''fruit.py: Validación del gps de un conjunto de imagenes con el gps de plantas.
    *El script recibe un modelo preentrenado, un video al cual se le ejecutara la inferencia y recibe el path donde se guardara el video de la inferencia (es opcional). El script ejecuta la inferencia sobre el video, muestra el video por pantalla y luego (si se paso el path) se guarda el video en el path proveido.

    *Para ello, recibe 10 argumentos (3 obligatorios y los demas optionales), el modelo preentrenado, el path del video a inferir, el path (con el nombre) del video de la inferencia (Optional),la probabilidad mínima para filtrar las detecciones débiles(Optional,default=0.4),
    fotogramas omitidos entre las detecciones(Optional,default=5),tamaño de la imagen de la inferencia(Optional,default=1024),parametros de yolo(Optional, default= 0.25, default=0.45),número máximo de detecciones por cuadro(Optional,defual=1000) y si quieres que use la CPU o GPU. Nota: el script recibe un unico modelo y un unico video.

    Para ejecutarlo seguir los siguientes pasos:
    1.  Para correr el script, ingrese a la carpeta 'code/fruit_counter/' dentro del repositorio de winter-variables. Puede ejecutar el archivo runfrutillas.sh, pero previamente debera editar el archivo con los parametros correctos, y ademas darle permisos de ejecucion al archivo con 'chmod +x runfrutillas.sh'. La otra opcion de ejecucion, es ejecutar directamente el archivo  fruit.py , tambien se encuentra en la carpeta 'code/fruit_counter/', y si le pasa el parametro -h tendra una informacion detallada de cada parametro.
    2. Una vez ejecutado el script, el mismo mostrara el video con la deteccion de frutillas.
        ''')
        parser.add_argument("-i", "--input", type=str,required=True, help="path to optional input video file")
        parser.add_argument("-o", "--output", type=str, help="path to optional output video file")
        parser.add_argument("-c", "--confidence", type=float, default=0.4,help="minimum probability to filter weak detections")
        parser.add_argument("-s", "--skip-frames", type=int, default=5, help="# of skip frames between detections")
        parser.add_argument("-y","--imgsz",type=int,default=1024, help="size of image of inference")
        parser.add_argument("-k","--conf-thres",type=float,default=0.25, help="")
        parser.add_argument("-l","--iou-thres",type=float, default=0.45, help="")
        parser.add_argument("-p","--max-det",type=int,default=1000, help="maximum number of detections per frame")
        parser.add_argument("-u","--is-gpu",type=str,required=True, help="if you want it to use CPU pass it 'cpu', otherwise pass it an empty string '' for GPU use.")
        args = parser.parse_args()
        return args

    args = vars(parse_opt())
    print(args)
    imgsz=args["imgsz"]
    conf_thres=args["conf_thres"]
    iou_thres=args["iou_thres"]
    max_det=args["max_det"]
    is_gpu=args["is_gpu"]
    confidence =args["confidence"]
    classes=None
    agnostic_nms=False
    half=False,  # use FP16 half-precision inference
    device = select_device(is_gpu) #change by '' for CUDA devices
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size


    print("[INFO] Starting the video..")
    vs = cv2.VideoCapture(args["input"])

    # initialize the video writer (we'll instantiate later if need be)
    writer = None

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=20, maxDistance=50)
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

    # start the frames per second throughput estimator
    fps = FPS().start()

    if config.Thread:
        vs = thread.ThreadingClass(config.url)

    # loop over frames from the video stream
    while True:
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        frame = vs.read()
        rgb = frame

        if frame is None:
            break

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                (W, H), True)

        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % args["skip_frames"] == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []
            # Llama al detector de circulos
            preds = demo(opt)


            # loop over the detections
            for pred in reversed(preds):
              center_x = pred[0]
              center_y = pred[1]
              radio = pred[2]
              conf = pred[3]
              classl = pred[4]
              startX, startY = center_x-radio,center_y-radio
              endX, endY = center_x+radio,center_y+radio
              if conf > args["confidence"]:
                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)
                      

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we have to add the centroid
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)


            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
                
        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


if __name__ == '__main__':
  #load options from argparse
  opt = opts().init()
  run(opt)