import cv2
import os
import csv
from datetime import datetime
from deepface import DeepFace

from utils.helpers import generar_nombre_archivo, asegurar_directorio

# Rutas
carpeta_rostros = "rostros_conocidos"
carpeta_registro = "registros"
archivo_registro = os.path.join(carpeta_registro, "registro.csv")
carpeta_desconocidos = "desconocidos"

# Asegurar carpetas
asegurar_directorio(carpeta_registro)
asegurar_directorio(carpeta_desconocidos)

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Cargar clasificador Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Cargar imágenes conocidas
rostros_conocidos = []
nombres_conocidos = []

for archivo in os.listdir(carpeta_rostros):
    if archivo.endswith(".jpg") or archivo.endswith(".png"):
        ruta = os.path.join(carpeta_rostros, archivo)
        rostro = cv2.imread(ruta)
        if rostro is not None:
            rostros_conocidos.append(rostro)
            nombres_conocidos.append(os.path.splitext(archivo)[0])

# Set para evitar duplicados por sesión
personas_registradas = set()

# Iniciar bucle
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in rostros:
        rostro = frame[y:y+h, x:x+w]
        nombre_persona = "Desconocido"

        try:
            for i, rostro_conocido in enumerate(rostros_conocidos):
                resultado = DeepFace.verify(rostro, rostro_conocido, enforce_detection=False)
                if resultado["verified"]:
                    nombre_persona = nombres_conocidos[i]
                    break
        except Exception as e:
            print("Error en reconocimiento:", e)

        # Enmarcar rostro con color según sea conocido o no
        color = (0, 255, 0) if nombre_persona != "Desconocido" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, nombre_persona, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Registro de persona si no se ha registrado ya
        if nombre_persona not in personas_registradas:
            hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(archivo_registro, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([nombre_persona, hora_actual])
            personas_registradas.add(nombre_persona)

        # Guardar imagen de rostro desconocido
        if nombre_persona == "Desconocido":
            rostro_guardado = frame[y:y+h, x:x+w]
            nombre_archivo = generar_nombre_archivo(base="desconocido")
            ruta_guardado = os.path.join(carpeta_desconocidos, nombre_archivo)
            cv2.imwrite(ruta_guardado, rostro_guardado)

    # Mostrar fecha y hora en pantalla
    hora_texto = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, hora_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Detector de Rostros", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
