import cv2
import numpy as np

# Now we will write code to capture the frame

cap = cv2.VideoCapture(0)

# Below is the text file containing the name of the predefined classes which can YOLO easily detect
classFile = 'coco.name.txt'

# A list to contain all the class name
classNames = []

# Storing the classnames from text file
with open (classFile,'r') as f:
    classNames = f.read().rstrip('\n').split('\n')

# print(classNames)
# print(len(classNames))

# Global variable for width and height.
wht = 320

# If the value is less than confThreshold, then the boundary box will not detect any class
confThreshold = .5

# The lower the value of nmsThreshold the less object will be detected in boundary boxes
nmsThreshold = .3

# the builtin YOLO model configuration file
modelConfiguration = r'C:\Users\abdul\PycharmProjects\pythonProject\yolov3.cfg'

# The weight file downloaded from YOLO website.
modelWeights = r'C:\Users\abdul\PycharmProjects\pythonProject\yolov3.weights'

# Deep neural network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
# Setting OPENCV as backend so that it runs properly in OPENCV
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# Using CPU to train
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Function to detect objects.
def findOjects(outputs,img):
    # Getting height, width and confidence from img
    hT, wT,cT = img.shape
    # Below is a list for getting the index of those boundary boxes which satisfies the condition
    # (If the confidence level of that particular box is greater
    # than the threshold value)
    bbox = []
    # Below is the list for storing the class id of those boundary boxes which satisfies the condition
    classIds = []
    # The cpnfidence values for the boundary boxes.
    confs = []
    
    for output in outputs:
        for det in output:
            # Please uncomment the following line to understand why I chose scores excluding first five values
            # print(scores)
            scores = det[5:]
            # Finding the index of maximum score
            classId = np.argmax(scores)
            # Finding the maximum confidence value
            confidence = scores[classId]
            if confidence > confThreshold:
                # Finding the weight and height
                w,h = int(det[2]*wT), int(det[3]*hT)
                # Finding the x and y
                x,y = int((det[0]*wT) - w/2) , int((det[1]*hT) - h/2)
                # Now as we have found x,y,w,h we need to append the values to bbox.
                bbox.append([x,y,w,h])
                # The class ids of chosen boundary boxes
                classIds.append(classId)
                # Confidence boxes of chosen classes
                confs.append(float(confidence))
    # cv2.dnn.NMSBoxes exclude the overlapping boundary boxes and return the indice of the remaining boundary boxes.
    indices_nms = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    # print(indices_nms)
    for i in indices_nms:
    #     i = i[0]
    #     Finding the accurate boundary box.
        box = bbox[i]
        # Finding x,y,w,h to detect object.
        x,y,w,h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        # To form rectangle around the objects
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        # To describe what the object is.
        cv2.putText(img, f'{classNames[classIds[i]]} {int(confs[i]*100)} %',(x+150,y), cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)

#         Infinite loop to detect objects until it is stopped. Please be careful while running it.
while True:
    # To read the image from video framerate using cv2.
    success, img = cap.read()
    # img = cv2.imread("luxury-car-model-toy-car-2101619.jpg", cv2.IMREAD_UNCHANGED)
    # Blob is a module for converting the image and processing it properly so that cv2 can read it and analyze it properly.
    blob = cv2.dnn.blobFromImage(img,1/255,(wht,wht),[0,0,0],1, crop = False)

    # Setting the actual input to opencv network from blob
    net.setInput(blob)
    # There are three layers in YOLOv3. We will get the name of the layers. But these will be indices
    layerNames = net.getLayerNames()

    #print(layerNames)
   # print(net.getUnconnectedOutLayers())
   #  To print the layers in readable form and save them in a list
    outputNames = net.getUnconnectedOutLayersNames()
    #print(outputNames)
    # Pass the layer names in network to get the results from each of the layers.
    outputs = net.forward(outputNames)
  #   You can uncomment the lines to get a better understanding
  #  print(len(outputs))
  #   print((outputs[0].shape))
  #   print((outputs[1]).shape)
  #   print((outputs[2].shape))
  #   print(outputs[0][0])
  # Below function will be used to find the object.
    findOjects(outputs,img)
    # To show the image
    cv2.imshow('image', img)
    # Delay
    cv2.waitKey(1)
     


