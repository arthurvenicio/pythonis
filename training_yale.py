import cv2
import os
import numpy as np
from PIL import Image

eigenface = cv2.face.EigenFaceRecognizer_create(40, 8000)
fisherface = cv2.face.FisherFaceRecognizer_create(3, 2000)
lbph = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 50)


def getImageWithId():
    path = [os.path.join('yalefaces/training', f)
            for f in os.listdir('yalefaces/training')]
    faces = []
    ids = []
    for image_path in path:
        imageFace = Image.open(image_path).convert('L')
        imageNP = np.array(imageFace, 'uint8')
        id = int(os.path.split(image_path)[
                 1].split(".")[0].replace("subject", ""))
        ids.append(id)
        faces.append(imageNP)

    return np.array(ids), faces


ids, faces = getImageWithId()

print("Training...")
eigenface.train(faces, ids)
eigenface.write('classifierEigenYale.yml')

fisherface.train(faces, ids)
fisherface.write('classifierFisherYale.yml')

lbph.train(faces, ids)
lbph.write('classifierLBPHYale.yml')

print("Training successful.")
