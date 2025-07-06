import os
import sys
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import time
import mediapipe as mp

# Agregar el directorio raíz del proyecto al sys.path
PROYECTO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROYECTO_RAIZ not in sys.path:
    sys.path.insert(0, PROYECTO_RAIZ)

from utils.helpers import cargar_rostros_conocidos


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Control de Presencia - Omniface")
        self.root.geometry("850x620")
        self.root.resizable(False, False)

        self.running = False
        # Inicializar detector de rostros
        self.mp_face = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        self.cap = None

        # Video frame
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(pady=10)

        # Botones
        self.btn_frame = ttk.Frame(self.root)
        self.btn_frame.pack()

        self.btn_start = ttk.Button(self.btn_frame, text="Iniciar Cámara", command=self.iniciar_camara)
        self.btn_start.grid(row=0, column=0, padx=10)

        self.btn_stop = ttk.Button(self.btn_frame, text="Detener Cámara", command=self.detener_camara)
        self.btn_stop.grid(row=0, column=1, padx=10)

        # Footer
        self.footer = ttk.Label(self.root, text="Proyecto de Reconocimiento Facial - IESTP JDS", font=("Segoe UI", 10))
        self.footer.pack(pady=10)

        # Carga de rostros
        self.rostros_conocidos = cargar_rostros_conocidos()

    def iniciar_camara(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.running = True
            threading.Thread(target=self.mostrar_video, daemon=True).start()

    def detener_camara(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')

    
    def mostrar_video(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Flip para efecto espejo (opcional)
            frame = cv2.flip(frame, 1)

            # Conversión para Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)

            # Dibujar detecciones
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    width, height = int(bbox.width * w), int(bbox.height * h)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Convertir a imagen para Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

            time.sleep(0.03)  # ~30 FPS


# Ejecutar
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
