import csv
import os
import cv2
import pandas as pd
from datetime import date, datetime

# Ruta del archivo CSV
RUTA_CSV = os.path.join("registros", "registro.csv")
RUTA_DESCONOCIDOS = os.path.join("desconocidos")

def generar_nombre_archivo(base="rostro", ext="jpg"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base}_{timestamp}.{ext}"

def asegurar_directorio(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def get_fecha_hora():
    ahora = datetime.now()
    fecha = ahora.strftime("%Y-%m-%d")
    hora = ahora.strftime("%H:%M:%S")
    return fecha, hora

def obtener_hora_fecha_actual():
    """Retorna fecha y hora actual como cadena"""
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def registrar_asistencia(nombre, fecha, hora, archivo='registros/registro.csv'):
    columnas_esperadas = ["Nombre", "Fecha", "Hora"]

    # Crear archivo si no existe o está mal formado
    if not os.path.exists(archivo):
        df = pd.DataFrame(columns=columnas_esperadas)
        df.to_csv(archivo, index=False)
    else:
        try:
            df = pd.read_csv(archivo)
            if not all(col in df.columns for col in columnas_esperadas):
                raise ValueError("Columnas faltantes")
        except Exception:
            print("[!] Archivo dañado. Se volverá a crear.")
            df = pd.DataFrame(columns=columnas_esperadas)
            df.to_csv(archivo, index=False)

    # Verificar duplicado
    df = pd.read_csv(archivo)
    duplicado = df[(df["Nombre"] == nombre) & (df["Fecha"] == fecha)]

    if duplicado.empty:
        nuevo_registro = {"Nombre": nombre, "Fecha": fecha, "Hora": hora}
        df = pd.concat([df, pd.DataFrame([nuevo_registro])], ignore_index=True)
        df.to_csv(archivo, index=False)
        print(f"[✓] Asistencia registrada: {nombre} - {fecha} {hora}")
    else:
        print(f"[•] Ya se registró asistencia para {nombre} en {fecha}")

# Opcion funcional sin GUI
def registrar_asistencia_sg(nombre):
    """Registra nombre, fecha y hora en archivo CSV, si no está ya registrado ese mismo día"""
    fecha_actual = date.today().isoformat()

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
        hora_actual = datetime.now().strftime("%H:%M:%S")
        with open(RUTA_CSV, mode="a", newline="", encoding="utf-8") as archivo:
            writer = csv.writer(archivo)
            writer.writerow([nombre, fecha_actual, hora_actual])
        print(f"[✔] Asistencia registrada: {nombre} - {fecha_actual} {hora_actual}")
    else:
        print(f"[i] {nombre} ya fue registrado hoy.")

def guardar_rostro_desconocido_sg(imagen):
    """Guarda imagen de rostro desconocido en carpeta con nombre único"""
    # Carpeta para rostros desconocidos
    CARPETA_DESCONOCIDOS = "rostros_desconocidos"

    if not os.path.exists(CARPETA_DESCONOCIDOS):
        os.makedirs(CARPETA_DESCONOCIDOS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"desconocido_{timestamp}.jpg"
    ruta_archivo = os.path.join(CARPETA_DESCONOCIDOS, nombre_archivo)

    try:
        cv2.imwrite(ruta_archivo, imagen)
        print(f"[💾] Rostro desconocido guardado en: {ruta_archivo}")
    except Exception as e:
        print(f"[!] Error al guardar rostro desconocido: {e}")

def cargar_rostros_conocidos():
    ruta = "rostros_conocidos"
    rostros = {}
    if not os.path.exists(ruta):
        os.makedirs(ruta)

    for archivo in os.listdir(ruta):
        if archivo.lower().endswith((".jpg", ".png")):
            nombre = os.path.splitext(archivo)[0]
            imagen = cv2.imread(os.path.join(ruta, archivo))
            rostros[nombre] = imagen
    return rostros

def guardar_rostro_desconocido(rostro, fecha, hora):
    if not os.path.exists(RUTA_DESCONOCIDOS):
        os.makedirs(RUTA_DESCONOCIDOS)

    nombre_archivo = f"desconocido_{fecha}_{hora.replace(':', '-')}.jpg"
    ruta_completa = os.path.join(RUTA_DESCONOCIDOS, nombre_archivo)
    cv2.imwrite(ruta_completa, rostro)

def verificar_duplicado(nombre, fecha):
    if not os.path.exists(RUTA_CSV):
        return False
    df = pd.read_csv(RUTA_CSV)
    return not df[(df["Nombre"] == nombre) & (df["Fecha"] == fecha)].empty
