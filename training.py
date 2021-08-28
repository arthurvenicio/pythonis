import cv2
import os 
import numpy as np

eigen_face = cv2.face.EigenFaceRecognizer_create() 
fisher_face = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


def getImageWithId():
    path = [os.path.join('photos', p) for p in os.listdir('photos')]
    faces = []
    ids = []

    for image_path in path:
        imageFace = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(image_path)[-1].split('.')[1])
        # print(id)
        ids.append(id)
        faces.append(imageFace)
        cv2.imshow("Face", imageFace)
        cv2.waitKey(10)
    return np.array(ids), faces


ids, faces = getImageWithId()
# print(faces)
print("Training...")

eigen_face.train(faces, ids)
eigen_face.write('classifierEigen.yml')

fisher_face.train(faces, ids)
fisher_face.write('classifierFisher.yml')

lbph.train(faces, ids)
lbph.write('classifierLBPH.yml')

print("Training successful.")

getImageWithId()
