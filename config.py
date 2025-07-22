# config.py

# Detector por defecto
DETECTOR_ACTUAL = 'mediapipe'

# Referencias a funciones actuales
_func_detectar = None
_func_dibujar = None

# Importa ambos detectores
from detectores import detector_mediapipe
from detectores import detector_retinaface

def seleccionar_detector(nombre):
    """Cambia el detector activo en tiempo real."""
    global DETECTOR_ACTUAL, _func_detectar, _func_dibujar

    if nombre == 'mediapipe':
        DETECTOR_ACTUAL = 'mediapipe'
        _func_detectar = detector_mediapipe.detectar_rostros_mediapipe
        _func_dibujar = detector_mediapipe.dibujar_rostros
    elif nombre == 'retinaface':
        DETECTOR_ACTUAL = 'retinaface'
        _func_detectar = detector_retinaface.detectar_rostros_retinaface
        _func_dibujar = detector_retinaface.dibujar_rostros
    else:
        raise ValueError(f"Detector desconocido: {nombre}")

# Inicializa el detector por defecto al cargar el módulo
seleccionar_detector(DETECTOR_ACTUAL)

def detectar_rostros(frame):
    return _func_detectar(frame)

def dibujar_rostros(frame, rostros):
    return _func_dibujar(frame, rostros)
