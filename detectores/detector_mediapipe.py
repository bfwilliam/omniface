import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Inicializamos el detector
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def detectar_rostros(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = face_detection.process(img_rgb)
    
    rostros = []
    if resultados.detections:
        h, w, _ = frame.shape
        for det in resultados.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            ancho = int(bbox.width * w)
            alto = int(bbox.height * h)
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
