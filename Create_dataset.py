import cv2
import os


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.mkdir(dir)


face_id = input('enter your id ')
# Start capturing video
vid_cam = cv2.VideoCapture(0)
cascpath="haarcascade_frontalface_default.xml"
# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier(cascpath)

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
    elif count >= 50:
        print("Successfully Captured")
        break

# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()
