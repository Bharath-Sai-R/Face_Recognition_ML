import cv2
import numpy as np;
import xlwrite
import firebase
import time
import sys
from playsound import playsound

start = time.time()
period = 8
cascpath="haarcascade_frontalface_default.xml"
face_cas = cv2.CascadeClassifier(cascpath)
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('D:\miniproject\Trainer.yml')
flag = 0
id = 0
filename = 'filename'
dict = {
    'item1': 1
}
 #font = cv2.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 5, 1, 0, 1, 1)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray, 1.3, 7)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, conf = recognizer.predict(roi_gray)
        #print(id)
        if (conf<=50):
            if (id == 1):
                id = 'Bharath Sai R'
                if ((str(id)) not in dict):
                    print("Attendance Taken")
                    filename = xlwrite.output('attendance', 'class1', 1, id, 'yes')
                    dict[str(id)] = str(id)

            elif (id == 2):
                id = 'Binlet Binu'
                if ((str(id)) not in dict):
                    print("Attendance Taken")
                    filename = xlwrite.output('attendance', 'class1', 2, id, 'yes')
                    dict[str(id)] = str(id)

            elif (id == 3):
                id = 'Hemanth Reddy'
                if ((str(id)) not in dict):
                    print("Attendance Taken")
                    filename = xlwrite.output('attendance', 'class1', 3, id, 'yes')
                    dict[str(id)] = str(id)

            elif (id == 4):
                id = 'User4'
                if ((str(id)) not in dict):
                    print("Attendance Taken")
                    filename = xlwrite.output('attendance', 'class1', 3, id, 'yes')
                    dict[str(id)] = str(id)        

        else:
            #print("Can't Recognize")
            id = 'Unknown, can not recognize'
            flag = flag + 1
            break

        cv2.putText(img, str(id) + " " + str(conf), (x, y - 10), font, 0.55, (120, 255, 120), 1)
        # cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,(0,0,255));
    cv2.imshow('frame', img)
    # cv2.imshow('gray',gray);
    #if flag == 10:
        #playsound('transactionSound.mp3')
        #print("Transaction Blocked")
        #break
    #if time.time() > start + period:
        #break
    if cv2.waitKey(50) & 0xFF == 13:
        break

cap.release()
cv2.destroyAllWindows()
