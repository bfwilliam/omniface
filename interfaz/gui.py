import cv2
import tkinter as tk
from tkinter import Label, Button, StringVar, OptionMenu
from PIL import Image, ImageTk
import os
import sys

# --- Manejo de rutas ---
PROYECTO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROYECTO_RAIZ not in sys.path:
    sys.path.insert(0, PROYECTO_RAIZ)

import config

CAMERA_INDEX = 1  # Índice usado por DroidCam

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Reconocimiento Facial")
        self.root.geometry("800x650")

        # --- Selección del detector ---
        self.opciones_detectores = ["mediapipe", "retinaface"]
        self.seleccion_detector = StringVar(value=config.DETECTOR_ACTUAL)
        self.menu_detector = OptionMenu(root, self.seleccion_detector, *self.opciones_detectores, command=self.cambiar_detector)
        self.menu_detector.pack(pady=5)

        # --- Vista del video ---
        self.video_label = Label(self.root)
        self.video_label.pack()

        self.btn_salir = Button(self.root, text="Salir", command=self.cerrar)
        self.btn_salir.pack(pady=10)

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            print(f"No se pudo abrir cámara en índice {CAMERA_INDEX}")
            self.cap.release()
            self.root.destroy()
            return

        self.actualizar_video()

    def cambiar_detector(self, valor):
        config.seleccionar_detector(valor)
        print(f"[INFO] Detector cambiado a: {valor}")

    def actualizar_video(self):
        ret, frame = self.cap.read()
        if ret:
            rostros = config.detectar_rostros(frame)
            config.dibujar_rostros(frame, rostros)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.actualizar_video)

    def cerrar(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
