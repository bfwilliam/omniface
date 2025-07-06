import os
import cv2
import pandas as pd
from datetime import datetime

# Ruta del archivo CSV
RUTA_CSV = os.path.join("registros", "registro.csv")
RUTA_DESCONOCIDOS = os.path.join("desconocidos")

def get_fecha_hora():
    ahora = datetime.now()
    fecha = ahora.strftime("%Y-%m-%d")
    hora = ahora.strftime("%H:%M:%S")
    return fecha, hora

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
