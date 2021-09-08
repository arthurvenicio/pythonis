import cv2
import os
import numpy as np
from PIL import Image

detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# recognizer = cv2.face.EigenFaceRecognizer_create()
# recognizer.read("classifierEigenYale.yml")
# recognizer = cv2.face.FisherFaceRecognizer_create()
# recognizer.read("classifierFisherYale.yml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("classifierLBPHYale.yml")

totalHits = 0
percentageHits = 0.0
totalTrust = 0.0

paths = [os.path.join('yalefaces/test', f)
         for f in os.listdir('yalefaces/test')]
for pathImage in paths:
    imageFace = Image.open(pathImage).convert('L')
    imageFaceNP = np.array(imageFace, 'uint8')
    detectedFaces = detectorFace.detectMultiScale(imageFaceNP)
    for (x, y, l, a) in detectedFaces:
        foreseenId, trust = recognizer.predict(imageFaceNP)
        currentId = int(os.path.split(pathImage)[
            1].split(".")[0].replace("subject", ""))
        print(str(currentId) + " foi classificado como " +
              str(foreseenId) + " - " + str(trust))
        if foreseenId == currentId:
            totalHits += 1
            totalTrust += trust
        #cv2.rectangle(imagemFaceNP, (x, y), (x + l, y + a), (0, 0, 255), 2)
        #cv2.imshow("Face", imagemFaceNP)
        # cv2.waitKey(1000)
percentageHits = (totalHits / 30) * 100
totalTrust = totalTrust / totalHits
print("Hit percentage: " + str(percentageHits))
print("Total trust: " + str(totalTrust))
