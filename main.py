import cv2
import os
import numpy as np
from datetime import datetime
#from detectores.detector_haar import detectar_rostros
from detectores.detector_mediapipe import detectar_rostros_mediapipe as detectar_rostros

from reconocedor.reconocedor import reconocer_rostro
import pandas as pd

# Crear carpeta de registros si no existe
os.makedirs("registros", exist_ok=True)
archivo_csv = "registros/registro.csv"

# Crear archivo CSV si no existe
if not os.path.exists(archivo_csv):
    with open(archivo_csv, "w") as f:
        f.write("nombre,fecha,hora\n")

# Cargar cámara
cap = cv2.VideoCapture(0)
rostros_registrados = set()

print("[INFO] Iniciando sistema de reconocimiento OmniFace...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    caras = detectar_rostros(frame)

    for (x, y, w, h) in caras:
        rostro = frame[y:y+h, x:x+w]
        if rostro is not None and rostro.size > 0:
            # Guardar imagen temporal para pasar a DeepFace.find()
            temp_path = "temp.jpg"
            cv2.imwrite(temp_path, rostro)

            nombre = reconocer_rostro(temp_path)

            # Colores según reconocimiento
            color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)
            etiqueta = nombre

            # Dibujar rectángulo y nombre
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, etiqueta, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Registrar solo si no se ha registrado antes en la sesión
            if nombre not in rostros_registrados:
                hora = datetime.now().strftime("%H:%M:%S")
                fecha = datetime.now().strftime("%Y-%m-%d")
                with open(archivo_csv, "a") as f:
                    f.write(f"{nombre},{fecha},{hora}\n")
                rostros_registrados.add(nombre)

                if nombre == "Desconocido":
                    # Guardar rostro desconocido en subrcarpetas con fecha y hora
                    carpeta_dia = os.path.join("rostros_desconocidos", fecha)
                    os.makedirs(carpeta_dia, exist_ok=True)
                    nombre_archivo = f"Desconocido_{hora.replace(':', '-')}.jpg"
                    cv2.imwrite(os.path.join(carpeta_dia, nombre_archivo), rostro)
                else:
                    print("⚠️ Rostro detectado, pero no válido para recorte. Se omitió.")
            # Opcional: eliminar imagen temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Mostrar hora y fecha
    hora_actual = datetime.now().strftime("%H:%M:%S")
    fecha_actual = datetime.now().strftime("%Y-%m-%d")
    cv2.putText(frame, f"{fecha_actual} {hora_actual}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Mostrar la cámara
    cv2.imshow("Detector de Rostros", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
