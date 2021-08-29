import cv2

detector_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.read("classifierEigen.yml")
width, height = 220, 220
font = cv2.FONT_ITALIC

cam = cv2.VideoCapture(0)

while True:

    conected, image = cam.read()
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detecedFaces = detector_face.detectMultiScale(grayImage,
                                                  scaleFactor=1.5, minSize=(60, 60))

    for (x, y, w, h) in detecedFaces:

        imageFace = cv2.resize(grayImage[y:y + h, x:x + w], (width, height))
        center = x + w // 2, y + h // 2
        radius = round(w // 1.5)
        cv2.circle(image, center, radius, (0, 255, 127), 2)

        id, confidence = recognizer.predict(imageFace)

        name = ""
        if id == 1:
            name = 'Arthur'
        elif id == 2:
            name = 'Karol'
            
        cv2.putText(image, name, (x, y + (h + 30)), font, 2, (0, 255, 127))
        cv2.putText(image, str(confidence),
                    (x, y + (h + 50)), font, 1, (0, 255, 127))

    cv2.imshow("Webcam", image)
    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):
        break

cam.release()
cv2.destroyAllWindows()
