from datetime import datetime
import os

def generar_nombre_archivo(base="rostro", ext="jpg"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base}_{timestamp}.{ext}"

def asegurar_directorio(path):
    if not os.path.exists(path):
        os.makedirs(path)
