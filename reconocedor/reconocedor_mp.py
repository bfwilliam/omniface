import cv2
import numpy as np
import pickle
from deepface import DeepFace

# Cargar embeddings preentrenados
with open("reconocedor/embeddings.pkl", "rb") as f:
    base_embeddings = pickle.load(f)

# Límite para considerar un rostro como conocido (ajustable)
UMBRAL_DISTANCIA = 10  # Para Facenet, entre 8 y 12 suele ser adecuado

def reconocer_rostro_mp(frame, bbox):
    x, y, w, h = bbox
    rostro = frame[y:y+h, x:x+w]

    # Validar que el recorte no esté vacío
    if rostro is None or rostro.size == 0:
        print("[!] Recorte vacío, no se puede procesar.")
        return "Desconocido", (x, y, w, h)

    try:
        print("[🔍] Procesando rostro para embedding...")
        resultado = DeepFace.represent(img_path=rostro, model_name="Facenet", enforce_detection=False)

        if not resultado or "embedding" not in resultado[0]:
            print("[!] No se encontró embedding válido.")
            return "Desconocido", (x, y, w, h)

        embedding_actual = np.array(resultado[0]["embedding"])
        nombre_reconocido = "Desconocido"
        distancia_min = float("inf")

        # Comparar con embeddings registrados
        for nombre, embedding_registrado in base_embeddings.items():
            distancia = np.linalg.norm(embedding_actual - embedding_registrado)
            if distancia < distancia_min:
                distancia_min = distancia
                nombre_reconocido = nombre

        if distancia_min < UMBRAL_DISTANCIA:
            print(f"[✔] Reconocido: {nombre_reconocido} (distancia: {distancia_min:.2f})")
            return nombre_reconocido.capitalize(), (x, y, w, h)
        else:
            print(f"[✖] No coincidencia (distancia: {distancia_min:.2f})")
            return "Desconocido", (x, y, w, h)

    except Exception as e:
        print(f"[!] Error en reconocimiento: {e}")
        return "Desconocido", (x, y, w, h)
