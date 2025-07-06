from datetime import datetime
import os
import csv
import datetime
import cv2

# Ruta del archivo de registro
RUTA_CSV = "registros/registro.csv"
# Carpeta para rostros desconocidos
CARPETA_DESCONOCIDOS = "rostros_desconocidos"

def generar_nombre_archivo(base="rostro", ext="jpg"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base}_{timestamp}.{ext}"

def asegurar_directorio(path):
    if not os.path.exists(path):
        os.makedirs(path)

def registrar_asistencia(nombre):
    """Registra nombre, fecha y hora en archivo CSV, si no está ya registrado ese mismo día"""
    fecha_actual = datetime.date.today().isoformat()

    # Crear archivo si no existe
    if not os.path.exists(RUTA_CSV):
        with open(RUTA_CSV, mode="w", newline="", encoding="utf-8") as archivo:
            writer = csv.writer(archivo)
            writer.writerow(["Nombre", "Fecha", "Hora"])

    # Verificar si ya está registrado hoy
    with open(RUTA_CSV, mode="r", encoding="utf-8") as archivo:
        lector = csv.reader(archivo)
        registros = list(lector)
        ya_registrado = any(row[0] == nombre and row[1] == fecha_actual for row in registros[1:])

    if not ya_registrado:
        hora_actual = datetime.datetime.now().strftime("%H:%M:%S")
        with open(RUTA_CSV, mode="a", newline="", encoding="utf-8") as archivo:
            writer = csv.writer(archivo)
            writer.writerow([nombre, fecha_actual, hora_actual])
        print(f"[✔] Asistencia registrada: {nombre} - {fecha_actual} {hora_actual}")
    else:
        print(f"[i] {nombre} ya fue registrado hoy.")

def obtener_hora_fecha_actual():
    """Retorna fecha y hora actual como cadena"""
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def guardar_rostro_desconocido(imagen):
    """Guarda imagen de rostro desconocido en carpeta con nombre único"""
    if not os.path.exists(CARPETA_DESCONOCIDOS):
        os.makedirs(CARPETA_DESCONOCIDOS)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"desconocido_{timestamp}.jpg"
    ruta_archivo = os.path.join(CARPETA_DESCONOCIDOS, nombre_archivo)

    try:
        cv2.imwrite(ruta_archivo, imagen)
        print(f"[💾] Rostro desconocido guardado en: {ruta_archivo}")
    except Exception as e:
        print(f"[!] Error al guardar rostro desconocido: {e}")