import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()

fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


def getImageWithId():
    path = [os.path.join('photos', p) for p in os.listdir('photos')]
    faces = []
    ids = []

    for pathImage in path:
        imageFace = cv2.cvtColor(cv2.imread(pathImage), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(pathImage)[-1].split('.')[1])
        print(id)
        ids.append(id)
        faces.append(imageFace)
        # cv2.imshow("Face", imageFace)
        # cv2.waitKey(10)
    return np.array(ids), faces


ids, faces = getImageWithId()

print("Training...")

eigenface.train(faces, ids)
eigenface.write('classifierEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classifierFisher.yml')

lbph.train(faces, ids)
lbph.write('classifierLBPH.yml')

print("Training successful")

getImageWithId()
