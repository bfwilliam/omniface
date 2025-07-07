import os
import pickle
import numpy as np
from deepface import DeepFace

def generar_embedding_individual(nombre, rutas_imagenes):
    """
    Genera un embedding promedio para un usuario a partir de sus imágenes.
    """
    embeddings = []

    print(f"\n[🧠] Procesando imágenes para '{nombre}'...")

    for ruta in rutas_imagenes:
        try:
            representacion = DeepFace.represent(img_path=ruta, model_name="Facenet", enforce_detection=False)
            if representacion and isinstance(representacion, list):
                embedding = representacion[0].get("embedding")
                if embedding:
                    embeddings.append(np.array(embedding))
                    print(f"[✔] Embedding obtenido desde: {ruta}")
                else:
                    print(f"[!] Embedding vacío en: {ruta}")
            else:
                print(f"[!] No se detectó rostro en: {ruta}")
        except Exception as e:
            print(f"[✖] Error con {ruta}: {e}")

    if not embeddings:
        print(f"[❌] No se pudo generar embeddings para {nombre}.")
        return None

    embedding_promedio = np.mean(embeddings, axis=0)
    print(f"[✅] Embedding promedio generado para '{nombre}'")
    return embedding_promedio


def actualizar_embeddings(nombre, embedding_promedio, ruta_embeddings="reconocedor/embeddings.pkl"):
    """
    Guarda o actualiza el embedding promedio en el archivo .pkl
    """
    if embedding_promedio is None:
        print("[⚠️] Embedding inválido, no se guardará.")
        return False

    if os.path.exists(ruta_embeddings):
        with open(ruta_embeddings, "rb") as f:
            embeddings = pickle.load(f)
    else:
        embeddings = {}

    embeddings[nombre] = embedding_promedio

    with open(ruta_embeddings, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"[💾] Embedding de '{nombre}' guardado en {ruta_embeddings}")
    return True
