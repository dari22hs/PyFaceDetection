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

# Capture video from webcam / Capturar video desde la webcam
webcam = cv2.VideoCapture(0)
# Capture video from route / Capturar video desde una ruta
# webcam = cv2.VideoCapture('video.mp4')

# Iterate forever over frames / Iterar por siempre sobre los frames
while True:
    successful_Frame_read, frame = webcam.read()

    # Change to grayscale / Cambiar a escala de grises
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    # Assign coordinates and loop / Asignar coordenadas y ciclar
    for (x, y, w, h) in face_coordinates:
    # Draw rectangle around the face / Dibujar rectálngulo alrededor del rostro -> (image, coordinates, line color, line thickness)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 5)
          
    # Show the image / Muestra la imagen
    cv2.imshow("\tFACE DETECTOR", frame)

    # Wait 1ms and then continues the program / Espera hasta que se presione una tecla y luego continúa el programa
    key = cv2.waitKey(1)
    
    # Stop if Q or q is pressed / Detener si Q o q es presionada -> 81 and 113 because of the ASCII characters
    if key == 81 or key == 113:
        break

# Release videocapture object / Liberar el objeto de video
webcam.release()
print("Code completed!")
    