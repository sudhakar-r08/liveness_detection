# USAGE
# python liveness_demo.py --model liveness.model --le le.pickle --detector face_detector

# import the necessary packages

import argparse
import os
import pickle
import socket
import subprocess
import cv2
import imutils
import numpy as np
import requests
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="liveness.model", help="path to trained model")
ap.add_argument("-l", "--le", type=str, default="le.pickle", help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, default="face_detector",
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.75,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)
cap = cv2.VideoCapture()


# vs.open("rtsp://admin:admin@12345@192.168.1.102:554/Streaming/channels/1/")
# cap.open("http://192.168.1.6:8082/video.mjpg?q=30&fps=33&id=0.2729321831683187&r=1586790060214")

def get_id():
    return subprocess.Popen('wmic bios get serialnumber')
    #return subprocess.Popen('wmic bios get guid')


# update data to the api
def update_api(param):
    r = requests.post('http://localhost/liveness-websocket/api/liveness.php',
                      data={'SubjectId': '', 'Status': param})
    print(r.text)


def log_status(host, status):
    r = requests.post('http://localhost/liveness-websocket/api/log.php',
                      data={'HostName': host, 'Organization': 'XYZ', 'Status': status})
    print(r.json())


def liveness_detection(cap):
    cap.open(0)
    curr_frame = 0
    isReal = False
    # # loop over the frames from the video stream
    while cap.isOpened():
        # frameId = cap.get(1)
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 600 pixels
        rect, frame = cap.read()
        fps = round(cap.get(cv2.CAP_PROP_FPS) / 3)
        frame = imutils.resize(frame, width=1000)
        # myframe = frame
        # cv2.imshow("Liveness Demo", myframe)
        if rect != True:
            break
        if curr_frame % fps == 0:
            print(curr_frame)
            print(fps)

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > args["confidence"]:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face and extract the face ROI
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # ensure the detected bounding box does fall outside the
                    # dimensions of the frame
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    # extract the face ROI and then preproces it in the exact
                    # same manner as our training data
                    face = frame[startY:endY, startX:endX]
                    face = cv2.resize(face, (32, 32))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    # pass the face ROI through the trained liveness detector
                    # model to determine if the face is "real" or "fake"
                    preds = model.predict(face)[0]
                    j = np.argmax(preds)
                    if preds[j] > 0.7:
                        label = le.classes_[j]
                        # draw the label and bounding box on the frame
                        label = "{}: {:.4f}".format(label, preds[j])

                        if j == 0:
                            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                        2)
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                            update_api(False)
                        else:
                            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                        2)
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                            update_api(True)

            # show the output frame and wait for a key press
            cv2.imshow("Liveness Demo", frame)
        cv2.imshow("Liveness Demo", frame)
        key = cv2.waitKey(1) & 0xFF

        curr_frame += 1
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    cap.stop()


def check_validity(param):
    r = requests.get('http:url.com/t=' + param)
    json = r.json()
    if json["Status"]:
        log_status(param, "Logged In")
        liveness_detection(cap)
    else:
        log_status(param, "Expired")


print(get_id())
check_validity(socket.gethostname())
