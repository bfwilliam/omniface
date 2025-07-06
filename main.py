import cv2
import datetime
import os
import mediapipe as mp
import numpy as np
from reconocedor.reconocedor import reconocer_rostro
from utils.helpers import registrar_asistencia, obtener_hora_fecha_actual, guardar_rostro_desconocido

# Configuración
registrados_en_sesion = set()
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Crear carpeta para desconocidos si no existe
os.makedirs("rostros_desconocidos", exist_ok=True)

# Inicializar cámara
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = face_detection.process(frame_rgb)

        if resultados.detections:
            for deteccion in resultados.detections:
                bboxC = deteccion.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Ajuste de coordenadas para evitar recortes fuera de imagen
                x, y = max(0, x), max(0, y)
                w, h = min(iw - x, w), min(ih - y, h)

                nombre = reconocer_rostro(frame, (x, y, w, h))

                # Mostrar nombre, fecha y hora
                hora_fecha = obtener_hora_fecha_actual()
                color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{nombre}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, hora_fecha, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Registrar solo una vez por sesión
                if nombre not in registrados_en_sesion:
                    registrados_en_sesion.add(nombre)
                    registrar_asistencia(nombre)
                    if nombre == "Desconocido":
                        guardar_rostro_desconocido(frame[y:y+h, x:x+w])

        cv2.imshow("Detector de Presencia", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

cap.release()
cv2.destroyAllWindows()
