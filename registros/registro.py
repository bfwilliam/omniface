# registros/registro.py

import os
import pandas as pd
from datetime import datetime

RUTA_REGISTRO = "registros/registro.csv"

def inicializar_csv():
    if not os.path.exists(RUTA_REGISTRO):
        df = pd.DataFrame(columns=["nombre", "fecha", "hora"])
        df.to_csv(RUTA_REGISTRO, index=False)

def registrar_presencia(nombre):
    ahora = datetime.now()
    fecha = ahora.strftime("%Y-%m-%d")
    hora = ahora.strftime("%H:%M:%S")

    # Si es "Desconocido", registrar igual con timestamp único
    if "Desconocido" in nombre:
        nombre = f"{nombre}_{fecha}_{hora.replace(':', '-')}"  # Para evitar duplicados

    # Cargar archivo y verificar si ya está (solo si no es "Desconocido")
    if os.path.exists(RUTA_REGISTRO):
        df = pd.read_csv(RUTA_REGISTRO)
        if "Desconocido" not in nombre:
            if not df[(df["nombre"] == nombre) & (df["fecha"] == fecha)].empty:
                return  # Ya registrado hoy

    nuevo = pd.DataFrame([[nombre, fecha, hora]], columns=["nombre", "fecha", "hora"])
    nuevo.to_csv(RUTA_REGISTRO, mode='a', header=not os.path.exists(RUTA_REGISTRO), index=False)
