import numpy as np 
import argparse
import cv2 as cv
import os 


args = argparse.ArgumentParser()

args.add_argument("-i", "--image", required=True, help="path to image")
args.add_argument("-c", "--confidence", type=float, default=0.2, help="min float confidence")
arguments = vars(args.parse_args())

resources = 'resources/'
modelpath = os.path.join(resources,os.listdir(resources)[1])
protopath = os.path.join(resources,os.listdir(resources)[0])
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0,255,size=(len(CLASSES),3))

net = cv.dnn.readNetFromCaffe(protopath,modelpath)

image = cv.imread(arguments['image'])
(h,w) = image.shape[:2]
blob = cv.dnn.blobFromImage(cv.resize(image,(300,300)), 0.007843, (300,300),127.5)
net.setInput(blob)
detections = net.forward()

for i in np.arange(0,detections.shape[2]):
    confidence = detections[0,0,i,2]
    if confidence > 0.2:
        idx = int(detections[0,0,i,1])
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startX,startY,endX,endY) = box.astype("int")
        print(confidence*100)
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        print("detected {}".format(label))
        cv.rectangle(image, (startX, startY), (endX, endY),COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv.putText(image, label, (startX, y),cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


cv.imshow('output',image)
cv.waitKey(0)
cv.destroyAllWindows()