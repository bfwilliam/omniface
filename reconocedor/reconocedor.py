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

def reconocer_rostro(cara, carpeta_rostros='rostros_conocidos', umbral=0.35):
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