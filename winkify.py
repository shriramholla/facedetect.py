# Required Modules
import cv2, time

# Cascade for Face Detection
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Cascade for Eye Detection
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Start Webcam
video = cv2.VideoCapture(0)

# While loop to keep video running

while True:
    # Obtain Frame from video
    check, frame = video.read()
    faces = None
    eyes = None

    # Grayscale Image used to detect face and eyes
    grayScaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Obtain Co-ordinates of face
    faces = cascade.detectMultiScale(grayScaleImage, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h in faces:

        # Construct Rectangle over face
        frame = cv2.rectangle(frame,(x,y),(x+h,y+w),(0,0,255),3)

        # Obtains co-ordinates of eyes
        eyes = eyeCascade.detectMultiScale(grayScaleImage, scaleFactor=1.1, minNeighbors=2)
        for x1, y1, w1, h1 in eyes:

            # Construct Rectangle over eyes
            frame = cv2.rectangle(frame,(x1,y1),(x1+h1,y1+w1),(0,255,0),3)

    # Display Frame
    cv2.imshow("Edited", frame)

    # Check for Blink
    if eyes is not None and len(eyes)==2:

        # Sleeps for 0.3s and checks if only one eye is detected

        time.sleep(0.3)
        check, frame = video.read()
        grayScaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Obtains new co-ordinates for eyes

        eyes = eyeCascade.detectMultiScale(grayScaleImage,scaleFactor=1.1, minNeighbors= 2)

        if len(eyes)==1:
            chk, capturedImage = video.read()
            cv2.imshow("Captured Image", capturedImage)

    # Exits Program is "Q" is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
