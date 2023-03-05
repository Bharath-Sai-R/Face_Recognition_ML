import tkinter as tk
import os,cv2;
import numpy as np
from PIL import Image;
from tkinter import messagebox
from datetime import datetime;
import pandas as pd
import cv2
import os
import cv2
import numpy as np
import xlwrite
import firebase
import time
import sys
from playsound import playsound

cascpath="haarcascade_frontalface_default.xml"
detector= cv2.CascadeClassifier(cascpath)
recognizer = cv2.face.LBPHFaceRecognizer_create()

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.mkdir(dir)

def take_img():
    ID = txt.get()
    Name = txt2.get()
    if(not(ID)):
        messagebox.showinfo(title="Invalid",message="Enter Id Value")
        return False
    if(not(Name)):
        messagebox.showinfo(title="Invalid",message="Enter Name")
        return False
    face_id = ID
    # Start capturing video
    vid_cam = cv2.VideoCapture(0)
    cascpath="haarcascade_frontalface_default.xml"
    # Detect object in video stream using Haarcascade Frontal Face
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize sample face image
    count = 0

    dir="D:/miniproject/dataset"
    #cdir="User"+str(face_id)
    #dir=os.path.join(pdir,cdir)
    assure_path_exists(dir)
    #os.mkdir(dir)
    os.chdir(dir)


    # Start looping
    while (True):
        # Capture video frame
        ret, image_frame = vid_cam.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

        # Detect frames of different sizes, list of faces rectangles
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # Loops for each faces
        for (x, y, w, h) in faces:
            # Crop the image frame into rectangle
            cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Increment sample face image
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            # Display the video frame, with bounded rectangle on the person's face
            cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
        if cv2.waitKey(100) & 0xFF == 13:
            break

        # If image taken reach 50, stop taking video
        elif count >= 100:
            messagebox.showinfo(title="Success",message="Successfully Captured")
            break

    # Stop video
    vid_cam.release()

    # Close all started windows
    cv2.destroyAllWindows()

    faces,Ids = getImagesAndLabels('D:\miniproject\dataset')
    s = recognizer.train(faces, np.array(Ids))
    #print("Successfully trained")
    messagebox.showinfo(title="Success",message="Successfully trained")
    recognizer.save('D:\miniproject\Trainer.yml')

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

def atend():
    start = time.time()
    period = 8
    #name=txt2.get()
   # cascpath="haarcascade_frontalface_default.xml"
    face_cas =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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


def opreg():
    os.system('start excel.exe D:\miniproject\database'+'attendance'+str(datetime.now().date())+'.xls')
    messagebox.showinfo(title="Success",message="Successfully Captured")


window = tk.Tk()
window.title("Attendance Management System")
window.geometry('1280x720')
window.configure()

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()
window.protocol("WM_DELETE_WINDOW", on_closing)



Notification = tk.Label(window, text="All things good", bg="Green", fg="white", width=15, height=3)

lbl = tk.Label(window, text="Enter id", width=20, height=2, fg="black", font=('times', 20, ' bold '))
lbl.place(x=200, y=200)

def testVal(inStr,acttyp):
    if acttyp == '1': #insert
        if not inStr.isdigit():
            return False
    return True
	
	
message = tk.Label(window, text="Attendence Management System", bg="magenta", fg="black", width=50,
                   height=3, font=('times', 30, ' bold '))

message.place(x=80, y=20)	

txt = tk.Entry(window, validate="key", width=20,  fg="black")
txt['validatecommand'] = (txt.register(testVal),'%P','%d')	
txt.place(x=550, y=210)

lbl2 = tk.Label(window, text="Enter Name", width=20, fg="black",  height=2, font=('times', 20, ' bold '))
lbl2.place(x=200, y=300)

txt2 = tk.Entry(window, width=20, fg="black")
txt2.place(x=550, y=310)

takeImg = tk.Button(window, text="Register",command=take_img,fg="black", bg="blue"  ,width=10  ,height=3, activebackground = "Red" ,font=('times', 20, ' bold '))
takeImg.place(x=200, y=500)

trainImg = tk.Button(window, text="Take Attendance",fg="black",command=atend ,bg="green"  ,width=20  ,height=3, activebackground = "Red",font=('times', 20, ' bold '))
trainImg.place(x=490, y=500)


opatt = tk.Button(window , text="List Attendance",fg="black", command=opreg ,bg="red"  ,width=20  ,height=3, activebackground = "Red",font=('times', 20, ' bold '))
opatt.place(x=890,y=500)


window.mainloop()