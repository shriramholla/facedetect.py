import cv2

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)
while True:
    check, frame = video.read()
    grayScaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(grayScaleImage, scaleFactor=1.23, minNeighbors=5)
    for x, y, w, h in faces:
        img = cv2.rectangle(frame,(x,y),(x+h,y+w),(0,0,255),3)
    cv2.imshow("Edited", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
