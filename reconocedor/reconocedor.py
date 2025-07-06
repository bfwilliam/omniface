import os
from deepface import DeepFace

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
