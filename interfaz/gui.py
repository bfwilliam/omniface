import os
import sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
import datetime

# Agregar el directorio raíz del proyecto al sys.path
PROYECTO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROYECTO_RAIZ not in sys.path:
    sys.path.insert(0, PROYECTO_RAIZ)
from detectores.detector_mediapipe import detectar_rostros_mediapipe
from utils.helpers import get_fecha_hora, registrar_asistencia, verificar_duplicado
from reconocedor.reconocedor import reconocer_rostro_gui, reconocer_rostro_mp

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Omniface - Control de Presencia")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        self.label_video = tk.Label(root)
        self.label_video.pack()

        self.contador_frames = 0
        self.procesar_frame = True

        self.actualizar_video()

        self.root.protocol("WM_DELETE_WINDOW", self.cerrar)

    def actualizar_video(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.root.after(10, self.actualizar_video)
            return

        nombre, color = "Desconocido", (0, 0, 255)
        bbox = None

        if self.procesar_frame:
            resultados = detectar_rostros_mediapipe(frame)
            if resultados:
                for rostro in resultados:
                    ombre, bbox = reconocer_rostro_gui(frame, rostro)
                    if bbox is None:
                        continue  # ignora si no hay bbox válido
                    if nombre is None:
                        nombre = "Desconocido"
                    color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)

                    if bbox:
                        x, y, w, h = bbox
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        fecha, hora = get_fecha_hora()
                        if not verificar_duplicado(nombre, fecha, hora):
                            registrar_asistencia(nombre, fecha, hora)

        # Mostrar fecha y hora
        fecha, hora = get_fecha_hora()
        cv2.putText(frame, f"{fecha} {hora}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Convertir para Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.label_video.imgtk = imgtk
        self.label_video.configure(image=imgtk)

        self.procesar_frame = not self.procesar_frame  # Procesa cada 2 frames
        self.root.after(30, self.actualizar_video)     # Control de velocidad

    def cerrar(self):
        self.cap.release()
        self.root.destroy()

# Ejecutar app
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()