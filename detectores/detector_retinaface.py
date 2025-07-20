import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Inicializamos el detector de rostros con insightface
app = FaceAnalysis(providers=['CPUExecutionProvider'])  # Puedes usar 'CUDAExecutionProvider' si tienes GPU
app.prepare(ctx_id=0, det_size=(640, 640))

def detectar_rostros_retinaface(frame):
    rostros = []
    h, w, _ = frame.shape

    # Convertir a RGB (insightface trabaja con RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detectar rostros
    faces = app.get(img_rgb)
    
    for face in faces:
        # Coordenadas absolutas del bounding box
        x1, y1, x2, y2 = [int(coord) for coord in face.bbox]
        x = max(0, x1)
        y = max(0, y1)
        ancho = x2 - x1
        alto = y2 - y1
        rostros.append((x, y, ancho, alto))

    return rostros

def dibujar_rostros(frame, rostros, nombres=None):
    for i, (x, y, w, h) in enumerate(rostros):
        color = (0, 255, 0)  # Verde por defecto
        if nombres and i < len(nombres) and "Desconocido" in nombres[i]:
            color = (0, 0, 255)  # Rojo para desconocido

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if nombres and i < len(nombres):
            cv2.putText(frame, nombres[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
