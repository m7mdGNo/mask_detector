import cv2
import mediapipe as mp
import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")


mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
path = 'C:/Users/m7mde/Desktop/mask_model/images/'
cap = cv2.VideoCapture('test1.mp4')
name = 1
with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        success, img = cap.read()
        img = cv2.resize(img,(1280,720))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if results.detections:
            for id, detection in enumerate(results.detections):
                bBox = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                (x,y,s,k) = boundBox
                face = img[y:y+k,x:x+s]
                try:
                    face = cv2.resize(face,(200,200))
                except:
                    pass
                face = np.expand_dims(face,axis=0)
                val = model.predict(face)
                font = cv2.FONT_HERSHEY_SIMPLEX
                val = np.argmax(val)
                if val ==0:
                    cv2.putText(img, 'no mask', (boundBox[0],boundBox[1] - 20), font, 1, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.rectangle(img,(boundBox[0],boundBox[1]),(boundBox[0]+boundBox[2],boundBox[1]+boundBox[3]),(0,0,255),3)
                else:
                    cv2.putText(img, 'mask', (boundBox[0],boundBox[1] - 20 ), font, 1, (0, 255 ,0), 2, cv2.LINE_4)
                    cv2.rectangle(img,(boundBox[0],boundBox[1]),(boundBox[0]+boundBox[2],boundBox[1]+boundBox[3]),(0,255,0),3)
        cv2.imwrite(f'frames/{name}.png',img)
        name += 1
        
        cv2.imshow('Face Detection', img)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()