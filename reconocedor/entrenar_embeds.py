# reconocedor/entrenar_embeds.py

import os
import cv2
import pickle
from deepface import DeepFace

def generar_embeddings(carpeta_rostros="rostros_conocidos", archivo_salida="reconocedor/embeddings.pkl"):
    base_embeddings = {}

    for archivo in os.listdir(carpeta_rostros):
        if archivo.endswith(".jpg") or archivo.endswith(".png"):
            ruta = os.path.join(carpeta_rostros, archivo)
            nombre = os.path.splitext(archivo)[0]  # Ej. bryan.jpg -> bryan

            try:
                embedding = DeepFace.represent(img_path=ruta, model_name='Facenet')[0]["embedding"]
                base_embeddings[nombre] = embedding
                print(f"[✔] Procesado: {nombre}")
            except Exception as e:
                print(f"[✖] Error en {archivo}: {e}")

    with open(archivo_salida, "wb") as f:
        pickle.dump(base_embeddings, f)
    print(f"\n[💾] Embeddings guardados en {archivo_salida}")

if __name__ == "__main__":
    generar_embeddings()
