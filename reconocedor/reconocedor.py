import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace
from scipy.spatial.distance import cosine

def cargar_rostros_conocidos(directorio='rostros_conocidos'):
    """
    Carga las imágenes de personas conocidas desde el directorio.
    """
    rostros = []
    for archivo in os.listdir(directorio):
        if archivo.lower().endswith(('.jpg', '.png', '.jpeg')):
            nombre = os.path.splitext(archivo)[0]
            ruta = os.path.join(directorio, archivo)
            rostros.append({'nombre': nombre, 'imagen': ruta})
    return rostros

def reconocer_uno(rostro, lista_referencia, umbral=0.3):
    """
    Compara un rostro con la base de rostros conocidos.
    Devuelve el nombre si lo reconoce, sino 'Desconocido'.
    """
    for persona in lista_referencia:
        try:
            resultado = DeepFace.verify(rostro, persona['imagen'], enforce_detection=False)
            if resultado["verified"] and resultado["distance"] < umbral:
                return persona["nombre"]
        except Exception:
            continue
    return "Desconocido"

def reconocer_rostro(cara, carpeta_rostros='rostros_conocidos', umbral=0.4):
    """
    Compara una imagen de rostro con todos los rostros conocidos usando DeepFace.find().
    Retorna el nombre del rostro si se encuentra por debajo del umbral de similitud.
    """
    try:
        resultados = DeepFace.find(img_path=cara, db_path=carpeta_rostros, enforce_detection=False, silent=True)
        
        if resultados and not resultados[0].empty:
            top_result = resultados[0].iloc[0]
            distancia = top_result['distance']
            nombre = top_result['identity'].split("/")[-1].split(".")[0]

            if distancia < umbral:
                return nombre
    except Exception as e:
        print(f"Error en reconocimiento: {e}")
    
    return "Desconocido"

# Actualizacion de reconocedor

# Umbral de similitud: cuanto menor, más estricto (ajustable)
UMBRAL_SIMILITUD = 0.4

# Cargar los embeddings precomputados
with open("reconocedor/embeddings.pkl", "rb") as f:
    base_embeddings = pickle.load(f)
    
def reconocer_rostro(frame, rostro_detectado):
    x, y, w, h = rostro_detectado
    rostro = frame[y:y+h, x:x+w]

    try:
        embedding_actual = DeepFace.represent(img_path=rostro, model_name="Facenet", enforce_detection=False)[0]["embedding"]
    except Exception as e:
        print(f"[!] Error al generar embedding: {e}")
        return "Error"

    similitudes = {}
    for nombre, embedding_guardado in base_embeddings.items():
        distancia = cosine(embedding_actual, embedding_guardado)
        similitudes[nombre] = distancia

    # Buscar la persona más parecida
    nombre_predicho = "Desconocido"
    if similitudes:
        nombre_similar = min(similitudes, key=similitudes.get)
        distancia_min = similitudes[nombre_similar]

        if distancia_min < UMBRAL_SIMILITUD:
            nombre_predicho = nombre_similar
        else:
            print(f"[?] Ninguna coincidencia aceptable (distancia mínima: {distancia_min:.2f})")

    return nombre_predicho