import cv2
import numpy as np

# Método para o software ativar a WebCam e detectar a face
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Método para o software conseguir detectar os olhos
eye_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")

# Variável para o software conseguir captura com a WebCam
cam = cv2.VideoCapture(0)

# Variável sample recebe as amostras de captura
sample = 1
num_samples = 25

# Id para diferenciar os rostos detectados
num_id = input('Please insert your id: ')

# Redimensionar as imagens com largura e altura fixos para não ocorrer erro de detectação facial
width = 220
height = 220

while True:
    conected, image = cam.read()

    # Variável que converte a imagem para escala de cinza
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Variável para definir a imagem que será analisada
    detecedFaces = classifier.detectMultiScale(
        grayImage, scaleFactor=1.5, minSize=(150, 150))

    # Estrura para contornar as faces dectadas pelo sofware com um círculo
    for (x, y, l, a) in detecedFaces:
        center = x + l // 2, y + a // 2
        radius = round(l // 1.5)
        cv2.circle(image, center, radius, (0, 255, 127), 2)

        # Bloco de variáveis para o software detectar os olhos
        region = image[y:y + a, x:x + l]
        region_eye_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        eyes_detected = eye_classifier.detectMultiScale(region_eye_gray)

        # Estrutura de repetição para contornar os olhos detectados
        for (ox, oy, ow, oh) in eyes_detected:
            cv2.rectangle(region, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 1)

            # Estrutura de condição para salvar as fotos capturadas pelo software, o salvamento é feio ao apertar a letra 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):

                # Estrutura condicional para captar a imagem caso a luminosidade atenda os requisitos
                if np.average(grayImage) > 110:
                    imageFace = cv2.resize(
                        grayImage[y:y + a, x:x + l], (width, height))
                    cv2.imwrite("photos/person." + str(num_id) +
                                "." + str(sample) + ".jpg", imageFace)
                    print("[foto " + str(sample) + "was captured with sucess]")
                    sample += 1

    # Método para o software ativar a WebCam
    cv2.imshow('Webcam', image)
    cv2.waitKey(1)

    # Estrutura de condição para caso o número de amostras ultrapassar o valor inicial, no caso 25
    if (sample >= num_samples + 1):
        break

print('All faces was captured with sucess!')

# Método para o software ativar a WebCam
cam.release()
cv2.destroyAllWindows()
