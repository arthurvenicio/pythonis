import cv2
import numpy as np

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")
cam = cv2.VideoCapture(0)
sample = 1
num_samples = 25
num_id = input('Please insert your id: ')
width = 220
height = 220

while True:
    conected, image = cam.read()

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detecedFaces = classifier.detectMultiScale(
        grayImage, scaleFactor=1.5, minSize=(150, 150))

    for (x, y, l, a) in detecedFaces:
        center = x + l // 2, y + a // 2
        radius = round(l // 1.5)
        cv2.circle(image, center, radius, (0, 255, 127), 2)

        region = image[y:y + a, x:x + l]
        region_eye_gray =  cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        eyes_detected = eye_classifier.detectMultiScale(region_eye_gray)

        for (ox, oy, ow, oh) in eyes_detected:
            cv2.rectangle(region, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 1)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if np.average(grayImage) > 110:
                    imageFace = cv2.resize(grayImage[y:y + a, x:x + l], (width, height))
                    cv2.imwrite("photos/person." + str(num_id) + "." + str(sample) + ".jpg", imageFace)
                    print("[foto " + str(sample) + "was captured with sucess]")
                    sample += 1

    cv2.imshow('Webcam', image)
    cv2.waitKey(1)
    if (sample >= num_samples + 1):
        break

print('All faces was captured with sucess!')
cam.realease()
cv2.destroyAllWindows()
