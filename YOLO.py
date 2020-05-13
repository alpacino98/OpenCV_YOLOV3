import cv2
import numpy as np
import os


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def output_layers(net):

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_pred(img, class_id, confidence, x, y, x_w, y_h):

    label= str(classes[class_id]) + '   ' + str(round(confidence, 3))

    color = (255, 0, 0)

    cv2.rectangle(img, (x,y), (x_w,y_h), color, 2)

    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


#video = cv2.VideoCapture('C:/Users/User/Downloads/11.avi')
#video.set(3, 640)
#video.set(4, 480)

with open('yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

boo = True
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
count = 0

cv_img = []

pathFiles = os.listdir("/users/alphapro/Desktop/DETECTION/Fruit_Recog/fruits-360_dataset/fruits-360/Test/")
for i in range(pathFiles.__len__()):
    path = "/users/alphapro/Desktop/DETECTION/Fruit_Recog/fruits-360_dataset/fruits-360/Test/" + pathFiles[i]
    images = os.listdir(path)
    for j in images:
        n = cv2.imread(path + "/" + j)
        cv_img.append(n)

video = cv2.VideoCapture(0)
while boo:
    check, image = video.read()
    image = rescale_frame(image , percent=100)
    """
    if(count < pic_path.__len__()):
        image =cv2.imread(pic_path[count])
        print(image)
        count += 1
    else:
        boo = False

    """

    '''
    carry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.zeros([carry.shape[0], carry.shape[1], 3])
    for i in range(3):
    image[:,:,i] = carry
    '''

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392




    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))



    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in (outs):
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box[0:4]
        draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    '''
    boo2 = True
    while boo2:
        key2 = cv2.waitKey(1)
        cv2.imshow('Object Detecction', image)
        if key2 == ord('s'):
            boo2 = False
        if key2 == ord('q'):
            boo = False
            break
        '''
    key2 = cv2.waitKey(1)
    cv2.imshow('Object Detecction', image)
    if key2 == ord('q'):
            boo = False    




video.release()
cv2.destroyAllWindows
