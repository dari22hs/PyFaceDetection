"""
Face Detector
This is a face detector program using OpenCV.
** Only for frontal face detection. **
-----------------------------------------------
Detector de rostros
Programa de detector de rostros usando OpenCV
** Solo para detección de rostros frontales. **
"""
import cv2
from random import randrange


# Load some pre-trained data on face frontals from opencv / Carga datos preentranados desde opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect / Elegir una imagen para detectar
img = cv2.imread('faces_test.jpg')

# Change to grayscale / Cambiar a escala de grises
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces / Detectar rostros
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Assign coordinates and loop / Asignar coordenadas y ciclar
for (x, y, w, h) in face_coordinates:
    # (x, y, w, h) = face_coordinates[0]
    # Draw rectangle around the face / Dibujar rectálngulo alrededor del rostro -> (image, coordinates, line color, line thickness)
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 5)

# Print coordenates / Imprimir coordenadas
print(face_coordinates)

# Show the image / Muestra la imagen
cv2.imshow("\tFACE DETECTOR", img)

# Wait until any key is pressed and then continues the program / Espera hasta que se presione una tecla y luego continúa el programa
cv2.waitKey()
print("Code completed!")
