# import the necessary packages
from tkinter.tix import Tree
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import pandas as pd
import threading
import argparse
import datetime
import imutils
import cv2
import torch
import os
import requests
import time
import asyncio
import websockets
import time
from playsound import playsound
from main_detection import detect
cam_pan_prev = 0
cam_tilt_prev = 0
horVal = 90
vertVal = 45
Iot_device_lock = False
Iot_device_lock_prev = False
Face_found_time = 0
Face_found_time_prev = time.time()
Save_data = []
# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
outputFrame_second = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
# vs = VideoStream(usePiCamera=1).start()
CamIPLink1 = 'http://192.168.1.100:81/stream'
CamIPLink2 = 'http://192.168.1.200:81/stream'

requests.get('http://192.168.1.100/control?var=framesize&val=8')
requests.get('http://192.168.1.200/control?var=framesize&val=8')
time.sleep(1)
requests.get('http://192.168.1.100/control?var=horizonservo&val=90')
requests.get('http://192.168.1.200/control?var=horizonservo&val=90')
time.sleep(1)
requests.get('http://192.168.1.100/control?var=verticalservo&val=90')
requests.get('http://192.168.1.200/control?var=verticalservo&val=90')
time.sleep(1)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index2.html")


def detect_faces():
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, outputFrame_second, lock

    # loop over frames from the video stream
    while cv2.waitKey(1):
        with lock:
            with torch.no_grad():
                outputFrame = detect(CamIPLink1)


def face_detected():
    playsound('mixkit-classic-alarm-995.wav')
    print(Face_found_time)
    print(Face_found_time_prev)


def detect_faces2():
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, outputFrame_second, lock

    # loop over frames from the video stream
    while cv2.waitKey(1):
        with lock:
            with torch.no_grad():
                outputFrame_second = detect(CamIPLink1)


def face_recognition(frame, frame_height, frame_width, camera_ip):
    global vertVal, horVal, cam_pan_prev, cam_tilt_prev, Face_found_time, Face_found_time_prev, Save_data
    temp_data = []
    timestamp = datetime.datetime.now()
    cv2.putText(frame, timestamp.strftime(
        "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    # Create a 4D blob from a frame.
    face_count = 0
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    # Run a model
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            face_count += 1
            Face_found_time = time.time()
            if (Face_found_time - Face_found_time_prev > 10):
                t4 = threading.Thread(target=face_detected)
                t4.daemon = True
                t4.start()
                temp_data = [face_count, timestamp]
                Save_data.append(temp_data)
                write_to_csv(Save_data)
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            x_center = int((x_left_bottom + x_right_top)/2)
            y_center = int((y_left_bottom + y_right_top)/2)

            x_img_center = int(frame_width/2)
            y_img_center = int(frame_height/2)

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

            t3 = threading.Thread(target=send_requests, args=(
                cam_pan, cam_pan_prev, cam_tilt, cam_tilt_prev, camera_ip,))
            t3.daemon = True
            t3.start()

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom),
                          (x_right_top, y_right_top), (0, 255, 0))
            cam_pan_prev = cam_pan
            cam_tilt_prev = cam_tilt
            Face_found_time_prev = Face_found_time

    cv2.putText(frame, "Number of faces: " + str(face_count), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    return frame


def send_requests(cam_pan, cam_pan_prev, cam_tilt, cam_tilt_prev, camera_ip):
    try:
        if(cam_pan_prev != cam_pan and abs(cam_pan_prev - cam_pan) >= 2):
            requests.get(
                'http://'+camera_ip+'/control?var=horizonservo&val=' + str(cam_pan))
    except:
        time.sleep(0.2)
    try:
        if(cam_tilt_prev != cam_tilt and abs(cam_tilt_prev - cam_tilt) >= 2):
            requests.get(
                'http://'+camera_ip+'/control?var=verticalservo&val=' + str(cam_tilt))
    except:
        time.sleep(0.2)


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
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


def generate2():
    # grab global references to the output frame and lock variables
    global outputFrame_second, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame_second is None:
                continue
            # encode the frame in JPEG format
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


async def test():
    try:
        async with websockets.connect('ws://172.16.0.218/ws') as websocket:
            await websocket.send("toggle")

            response = await websocket.recv()
            print(response)
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
        df = pd.DataFrame(lst, columns=['Number Of Faces', 'Time'])
        df.to_csv("test.csv", index=False)
    if file_state == "Not_Empty":
        df = pd.DataFrame(lst, columns=['Number Of Faces', 'Time'])
        df.to_csv("test.csv", mode="a", index=False, header=False)
    if file_state == "Nonexist":
        df = pd.DataFrame(lst, columns=['Number Of Faces', 'Time'])
        df.to_csv("test.csv", index=False)


@ app.route("/video_feed1")
def video_feed1():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@ app.route("/video_feed2")
def video_feed2():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate2(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    Face_found_time = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())
    # start a thread that will perform face detection
    t = threading.Thread(target=detect_faces)
    t.daemon = True
    t.start()
    t2 = threading.Thread(target=detect_faces2)
    t2.daemon = True
    t2.start()
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)
# release the video stream pointer
