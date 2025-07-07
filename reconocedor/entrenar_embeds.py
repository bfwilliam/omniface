import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace

def generar_embeddings(carpeta_rostros="rostros_conocidos", archivo_salida="reconocedor/embeddings.pkl"):
    base_embeddings = {}
    rostros_por_usuario = {}

    print("[📁] Escaneando carpeta de rostros...")

    # Agrupar imágenes por nombre (ej: bryan_1.jpg, bryan_2.jpg)
    for archivo in sorted(os.listdir(carpeta_rostros)):
        if archivo.endswith((".jpg", ".png")):
            nombre = os.path.splitext(archivo)[0].rsplit("_", 1)[0].lower()  # "Bryan_1" -> "bryan"
            if nombre not in rostros_por_usuario:
                rostros_por_usuario[nombre] = []
            ruta_completa = os.path.join(carpeta_rostros, archivo)
            rostros_por_usuario[nombre].append(ruta_completa)

    print(f"[ℹ️] Usuarios encontrados: {list(rostros_por_usuario.keys())}\n")

    for nombre, imagenes in rostros_por_usuario.items():
        embeddings = []
        print(f"[⏳] Procesando {nombre} ({len(imagenes)} imagen/es)...")

        for img_path in imagenes:
            print(f"   - {os.path.basename(img_path)}", end=" ")
            try:
                representacion = DeepFace.represent(img_path=img_path, model_name='Facenet', enforce_detection=False)
                if representacion and isinstance(representacion, list):
                    embedding = representacion[0].get("embedding")
                    if embedding:
                        embeddings.append(np.array(embedding))
                        print("[✔]")
                    else:
                        print("[✖] Embedding no encontrado")
                else:
                    print("[✖] Representación no válida")
            except Exception as e:
                print(f"[✖] Error: {e}")

        if embeddings:
            embedding_promedio = np.mean(embeddings, axis=0)
            base_embeddings[nombre] = embedding_promedio
            print(f"[✔] Embedding final generado para {nombre} (usando {len(embeddings)} muestra/s)\n")
        else:
            print(f"[!] No se generó embedding para {nombre} (rostros no detectados)\n")

    if base_embeddings:
        with open(archivo_salida, "wb") as f:
            pickle.dump(base_embeddings, f)
        print(f"[💾] Embeddings guardados exitosamente en: {archivo_salida}")
    else:
        print("[❌] No se guardó ningún embedding. Verifica las imágenes.")

if __name__ == "__main__":
    generar_embeddings()
