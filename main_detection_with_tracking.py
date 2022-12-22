import argparse
from random import randint
import time
from pathlib import Path
import socket
import cv2
import torch
import torch.backends.cudnn as cudnn
import os
import threading
import datetime
from numpy import random
from tkinter.tix import Tree
from imutils.video import VideoStream
from flask import Response
from flask import Flask, request
from flask import render_template
import pandas as pd
from playsound import playsound
import requests
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
# For SORT tracking
import skimage
from sort import *

lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
outputFrame = None
outputFrame_second = None

camIPLink = 'http://192.168.1.100:81/stream'
camIPLink2 = "http://192.168.1.200:81/stream"

Face_found_time = 0
Face_found_time_prev = time.time()
Mouse_posX_100 = 0
Mouse_posY_100 = 0
Mouse_posX_200 = 0
Mouse_posY_200 = 0
Save_data = []
cam_pan_prev = 0
cam_tilt_prev = 0
horVal = 90
vertVal = 90
tracking_lock_100 = False
tracking_lock_200 = False
try:
    requests.get('http://192.168.1.200/control?var=framesize&val=9')
    requests.get('http://192.168.1.100/control?var=framesize&val=9')
    time.sleep(1)
    requests.get('http://192.168.1.200/control?var=horizonservo&val=90')
    requests.get('http://192.168.1.100/control?var=horizonservo&val=90')
    time.sleep(1)
    requests.get('http://192.168.1.100/control?var=verticalservo&val=90')
    requests.get('http://192.168.1.200/control?var=verticalservo&val=45')
    time.sleep(1)
    requests.get("http://192.168.1.100/control?var=vflip&val=1")
    requests.get('http://192.168.1.200/control?var=vflip&val=1')
    time.sleep(1)
    requests.get("http://192.168.1.100/control?var=gainceiling&val=1")
    requests.get("http://192.168.1.200/control?var=gainceiling&val=1")
    time.sleep(1)
except:
    None

# Check if the mouse click point is in a detected square


def isInside(points, centroid):
    polygon = Polygon(points)
    centroid = Point(centroid)
    print(polygon.contains(centroid))
    return polygon.contains(centroid)


# ............................... Tracker Functions ............................
""" Random created palette"""
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

"""" Calculates the relative bounding box from absolute pixel values. """


def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


"""Function to Draw Bounding boxes"""


def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0]+box[2])/2), (int((box[1]+box[3])/2)))
        label = str(id) + ":" + names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)
    return img
# ..............................................................................


def send_requests(cam_pan, cam_pan_prev, cam_tilt, cam_tilt_prev, camera_ip):
    try:
        if(cam_pan_prev != cam_pan and abs(cam_pan_prev - cam_pan) >= 2):
            requests.get(
                'http://'+camera_ip+'/control?var=horizonservo&val=' + str(cam_pan))
    except:
        None
    try:
        if(cam_tilt_prev != cam_tilt and abs(cam_tilt_prev - cam_tilt) >= 2):
            requests.get(
                'http://'+camera_ip+'/control?var=verticalservo&val=' + str(cam_tilt))
    except:
        None


def face_detected():
    playsound('mixkit-classic-alarm-995.wav')


def detect(camIPLink, outputframe):
    global outputFrame, outputFrame_second, Face_found_time, Face_found_time_prev, horVal, vertVal, cam_pan_prev, cam_tilt_prev, tracking_lock_200, tracking_lock_100, Mouse_posX_100, Mouse_posY_100, Mouse_posX_200, Mouse_posY_200
    Number_of_faces = 0
    Send_signal = False
    Mouse_posX_100_prev = 0
    Mouse_posX_200_prev = 0
    Mouse_posY_100_prev = 0
    Mouse_posY_200_prev = 0
    x_center = 0
    y_center = 0
    with torch.no_grad():
        source, weights, imgsz = camIPLink, [
            'Pretrain\\yolov7.pt'], 640
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        # .... Initialize SORT ....
        # .........................
        sort_max_age = 5
        sort_min_hits = 2
        sort_iou_thresh = 0.35
        sort_tracker = Sort(max_age=sort_max_age,
                            min_hits=sort_min_hits,
                            iou_threshold=sort_iou_thresh)
        # .........................

        # Initialize
        set_logging()
        device = select_device("0")
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load(
                'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()

        # ........Rand Color for every trk.......
        rand_color_list = []
        for i in range(0, 5005):
            r = randint(0, 255)
            g = randint(0, 255)
            b = randint(0, 255)
            rand_color = (r, g, b)
            rand_color_list.append(rand_color)
    # .........................

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=False)[0]

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(
                pred, 0.58, 0.5, classes=[0], agnostic=False)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                    ), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(
                        dataset, 'frame', 0)

                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if len(det):
                    detections = 0
                    Face_found_time = time.time()
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    # ..................USE TRACK FUNCTION....................
                    # pass an empty array to sort
                    dets_to_sort = np.empty((0, 6))

                    # NOTE: We send in detected object class too
                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                        dets_to_sort = np.vstack((dets_to_sort,
                                                  np.array([x1, y1, x2, y2, conf, detclass])))

                    # Run SORT
                    tracked_dets = sort_tracker.update(dets_to_sort)
                    tracks = sort_tracker.getTrackers()
                    for track in tracks:
                        # color = compute_color_for_labels(id)
                        # draw colored tracks
                        if True:
                            [cv2.line(im0, (int(track.centroidarr[i][0]),
                                            int(track.centroidarr[i][1])),
                                      (int(track.centroidarr[i+1][0]),
                                       int(track.centroidarr[i+1][1])),
                                      rand_color_list[track.id], thickness=2)
                             for i, _ in enumerate(track.centroidarr)
                             if i < len(track.centroidarr)-1]

                    # Write results
                    if len(tracked_dets) > 0:
                        bbox_xyxy = tracked_dets[:, :4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        draw_boxes(im0, bbox_xyxy, identities,
                                   categories, names)
                        person_to_track = int(min(identities))
                        for i in tracked_dets:
                            if i[8] == person_to_track:
                                xyxy = i[:4]
                            else:
                                xyxy = np.array(
                                    [im0.shape[1]/2, im0.shape[0]/2, im0.shape[1]/2, im0.shape[0]/2])
                        x_left_bottom, y_left_bottom, x_right_top, y_right_top = [
                            int(i) for i in xyxy]
                        x_center = int((x_left_bottom + x_right_top)/2)
                        y_center = int((y_left_bottom + y_right_top)/2)
                        cv2.circle(im0, (x_center, y_center),
                                   1, (255, 0, 0), 5)
                    if source == "http://192.168.1.200:81/stream":
                        if(Mouse_posY_200 != Mouse_posY_200_prev or Mouse_posX_200_prev != Mouse_posX_200):
                            print("switching 200 lock")
                            tracking_lock_200 = not tracking_lock_200
                    if source == "http://192.168.1.100:81/stream":
                        if(Mouse_posY_100 != Mouse_posY_100_prev or Mouse_posX_100_prev != Mouse_posX_100):
                            tracking_lock_100 = not tracking_lock_100
                            print("switching 100 lock")
                # ........................................................
                    if (Face_found_time - Face_found_time_prev > 10):
                        Send_signal = True
                    Face_found_time_prev = Face_found_time
                    Number_of_faces = n.cpu().numpy()
                else:
                    Number_of_faces = 0
                    # Stream results
                if True:
                    x_img_center = int(im0.shape[1]/2)
                    y_img_center = int(im0.shape[0]/2)
                    if source == "http://192.168.1.100:81/stream" and tracking_lock_100 and x_center != -9999:

                        x_turn = int(x_center - x_img_center)
                        y_turn = int(y_center - y_img_center)
                        x_turn /= float(x_img_center)
                        y_turn /= float(y_img_center)
                        if(x_turn > 0 and x_turn > 0.1):
                            x_turn *= 8
                            horVal += x_turn

                        if(x_turn < 0 and x_turn < -0.1):
                            x_turn *= 8
                            horVal += x_turn
                        ###############################
                        if(y_turn > 0 and y_turn > 0.1):
                            y_turn *= 3
                            vertVal -= y_turn

                        if(y_turn < 0 and y_turn < -0.1):
                            y_turn *= -3
                            vertVal += y_turn
                        cam_pan = horVal
                        cam_tilt = vertVal

                        cam_pan = max(0, min(180, cam_pan))
                        cam_tilt = max(0, min(180, cam_tilt))

                        cam_pan = int(cam_pan)
                        cam_tilt = int(cam_tilt)
                        t7 = threading.Thread(target=send_requests, args=(
                            cam_pan, cam_pan_prev, cam_tilt, cam_tilt_prev, "192.168.1.100",))
                        t7.daemon = True
                        t7.start()

                        cam_pan_prev = cam_pan
                        cam_tilt_prev = cam_tilt
                    elif source == "http://192.168.1.200:81/stream" and tracking_lock_200 and x_center != -9999:
                        x_turn = int(x_center - x_img_center)
                        y_turn = int(y_center - y_img_center)

                        x_turn /= float(x_img_center)
                        y_turn /= float(y_img_center)

                        if(x_turn > 0 and x_turn > 0.1):
                            x_turn *= 8
                            horVal -= x_turn

                        if(x_turn < 0 and x_turn < -0.1):
                            x_turn *= -8
                            horVal += x_turn
                        ###############################
                        if(y_turn > 0 and y_turn > 0.1):
                            y_turn *= 2
                            vertVal -= y_turn

                        if(y_turn < 0 and y_turn < -0.1):
                            y_turn *= -2
                            vertVal += y_turn
                        cam_pan = horVal
                        cam_tilt = vertVal

                        cam_pan = max(0, min(180, cam_pan))
                        cam_tilt = max(0, min(180, cam_tilt))

                        cam_pan = int(cam_pan)
                        cam_tilt = int(cam_tilt)
                        t10 = threading.Thread(target=send_requests, args=(
                            cam_pan, cam_pan_prev, cam_tilt, cam_tilt_prev, "192.168.1.200",))
                        t10.daemon = True
                        t10.start()

                        cam_pan_prev = cam_pan
                        cam_tilt_prev = cam_tilt
                    timestamp = datetime.datetime.now()
                    cv2.circle(im0, (x_img_center, y_img_center),
                               1, (255, 0, 0), 5)
                    cv2.putText(im0, "Number of people detected: " + str(Number_of_faces), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    cv2.putText(im0, timestamp.strftime(
                        "%A %d %B %Y %I:%M:%S%p"), (10, im0.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    if Send_signal == True:
                        t4 = threading.Thread(target=face_detected)
                        t4.daemon = True
                        t4.start()
                        t8 = threading.Thread(target=send_alarm)
                        t8.daemon = True
                        t8.start()
                        temp_data = [Number_of_faces, timestamp.strftime(
                            "%A %d %B %Y %I:%M:%S%p")]
                        Save_data.append(temp_data)
                        write_to_csv(Save_data)
                    if source == "http://192.168.1.100:81/stream":
                        outputFrame = im0
                    else:
                        outputFrame_second = im0
            Mouse_posX_100_prev = Mouse_posX_100
            Mouse_posX_200_prev = Mouse_posX_200
            Mouse_posY_100_prev = Mouse_posY_100
            Mouse_posY_200_prev = Mouse_posY_200
            Send_signal = False
            Save_data.clear()


def send_alarm():
    try:
        clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the server

        clientSocket.connect(("192.168.1.150", 8888))

        # Send data to server

        data = "toggle"

        clientSocket.send(data.encode())

        # Receive data from server

        dataFromServer = clientSocket.recv(1024)

        print(dataFromServer)
    except:
        None


def check_if_file_empty_or_nonexist():
    try:
        if(os.stat("test.csv").st_size == 0):
            return "Empty"
        else:
            return "Not_Empty"
    except OSError:
        return "Nonexist"


def write_to_csv(lst):
    file_state = check_if_file_empty_or_nonexist()
    if file_state == "Empty":
        df = pd.DataFrame(lst, columns=['Number Of People Detected', 'Time'])
        df.to_csv("test.csv", index=False)
    if file_state == "Not_Empty":
        df = pd.DataFrame(lst, columns=['Number Of People Detected', 'Time'])
        df.to_csv("test.csv", mode="a", index=False, header=False)
    if file_state == "Nonexist":
        df = pd.DataFrame(lst, columns=['Number Of People Detected', 'Time'])
        df.to_csv("test.csv", index=False)


def generate(source):
    # grab global references to the output frame and lock variables
    global outputFrame, lock, outputFrame_second
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None or outputFrame_second is None:
                continue
            if source == 1:
                # encode the frame in JPEG format
                (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
                # ensure the frame was successfully encoded
                if not flag:
                    continue
            else:
                (flag, encodedImage) = cv2.imencode(".jpg", outputFrame_second)
                # ensure the frame was successfully encoded
                if not flag:
                    continue
        # yield the output frame in the byte format
        try:
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                  bytearray(encodedImage) + b'\r\n')
        except:
            print("Yield Exception")
            return


@app.route("/")
def index():
    # return the rendered template
    return render_template("index2.html")


@app.route("/video_feed1")
def video_feed1():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(1),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed2")
def video_feed2():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(2),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/web_mouse_click')
def web_mouse_click():
    global Mouse_posX_100, Mouse_posY_100, Mouse_posX_200, Mouse_posY_200
    Cam = request.args.get('no')
    if Cam == "1":
        Mouse_posX_100 = int(request.args.get('posX'))
        Mouse_posY_100 = int(request.args.get('posY'))
    elif Cam == "2":
        Mouse_posX_200 = int(request.args.get('posX'))
        Mouse_posY_200 = int(request.args.get('posY'))
    return "200"


if __name__ == '__main__':
    Face_found_time = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())
    t = threading.Thread(target=detect, args=(camIPLink, outputFrame,))
    t.daemon = True
    t.start()
    t1 = threading.Thread(target=detect, args=(
        camIPLink2, outputFrame_second,))
    t1.daemon = True
    t1.start()
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)
