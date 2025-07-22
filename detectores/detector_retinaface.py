import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Inicializamos el detector de rostros con insightface
app = FaceAnalysis(providers=['CPUExecutionProvider'])  # Usa 'CUDAExecutionProvider' si tienes GPU
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

        # Obtener landmarks (5 puntos)
        puntos = face.kps if hasattr(face, "kps") else []

        rostros.append({
            'bbox': (x, y, ancho, alto),
            'landmarks': puntos
        })

    return rostros

def dibujar_rostros(frame, rostros, nombres=None):
    for i, rostro in enumerate(rostros):
        x, y, w, h = rostro['bbox']
        color = (0, 255, 0)  # Verde por defecto
        if nombres and i < len(nombres) and "Desconocido" in nombres[i]:
            color = (0, 0, 255)  # Rojo para desconocido

        # Dibujar bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Dibujar nombre si existe
        if nombres and i < len(nombres):
            cv2.putText(frame, nombres[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Dibujar landmarks si existen
        puntos = rostro.get('landmarks', [])
        for (px, py) in puntos:
            cv2.circle(frame, (int(px), int(py)), 2, (0, 0, 255), -1)  # Rojo

    return frame
