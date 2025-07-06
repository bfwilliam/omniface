import pickle
import numpy as np
from deepface import DeepFace

# Cargar los embeddings previamente generados
with open("reconocedor/embeddings.pkl", "rb") as f:
    base_embeddings = pickle.load(f)

def reconocer_rostro_mp(frame, bbox, umbral=0.45):
    """
    Realiza reconocimiento facial utilizando DeepFace y embeddings preprocesados.
    
    Args:
        frame: Frame de la cámara (BGR)
        bbox: Tupla (x, y, w, h) de la cara detectada
        umbral: Umbral de similitud para considerar una coincidencia válida

    Returns:
        nombre (str): Nombre reconocido o "Desconocido"
        bbox (tuple): Bounding box para dibujar (x, y, w, h)
    """
    x, y, w, h = bbox
    rostro = frame[y:y+h, x:x+w]

    try:
        # Obtener embedding del rostro detectado (convertido a RGB)
        embedding_nuevo = DeepFace.represent(rostro, model_name='Facenet', enforce_detection=False)[0]["embedding"]

        # Comparar con base de embeddings
        similitudes = {}
        for nombre, emb_base in base_embeddings.items():
            distancia = np.linalg.norm(np.array(embedding_nuevo) - np.array(emb_base))
            similitudes[nombre] = distancia

        # Obtener el más cercano
        mejor_match = min(similitudes, key=similitudes.get)
        distancia = similitudes[mejor_match]

        if distancia < umbral:
            return mejor_match, (x, y, w, h)
        else:
            return "Desconocido", (x, y, w, h)

    except Exception as e:
        print(f"[!] Error en reconocimiento: {e}")
        return "Desconocido", (x, y, w, h)
