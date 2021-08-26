import cv2

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

while True:
    conected, image = cam.read()

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detecedFaces = classifier.detectMultiScale(grayImage, scaleFactor= 1.5, minSize=(100,100))

    for (x, y, l, a) in detecedFaces:
        
        center = x + l // 2 , y + a // 2
        radius = round(l // 1.5)
        cv2.circle(image ,center ,radius ,(0, 255, 128), 2)

    cv2.imshow('Webcam', image)
    cv2.waitKey(1)

cam.realease()
cv2.destroyAllWindows()