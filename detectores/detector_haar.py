import cv2

# Clasificador Haar para rostros frontales
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detectar_rostros(frame, scaleFactor=1.1, minNeighbors=5):
    """
    Detecta rostros en un frame y devuelve coordenadas (x, y, w, h)
    """
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gris, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    return rostros

def dibujar_rostros(frame, rostros, nombres=None):
    for i, (x, y, w, h) in enumerate(rostros):
        color = (0, 255, 0)  # Verde por defecto
        if nombres and i < len(nombres):
            if "Desconocido" in nombres[i]:
                color = (0, 0, 255)  # Rojo para desconocido
            nombre = nombres[i]
        else:
            nombre = ""

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if nombre:
            cv2.putText(frame, nombre, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
