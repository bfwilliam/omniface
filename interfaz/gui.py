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

from reconocedor.reconocedor import reconocer_rostro, reconocer_rostro_mp
from utils.helpers import (
    cargar_rostros_conocidos,
    registrar_asistencia,
    get_fecha_hora,
    verificar_duplicado,
    guardar_rostro_desconocido
)

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

        # Carga de rostros y registro
        self.rostros_conocidos = cargar_rostros_conocidos()
        self.rostros_detectados = set()

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
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = self.face_detection.process(frame_rgb)

        if resultados.detections:
            for detection in resultados.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                ancho = int(bbox.width * w)
                alto = int(bbox.height * h)

                # Recorte del rostro
                rostro = frame[y:y+alto, x:x+ancho]
                if rostro.size == 0:
                    continue

                nombre = reconocer_rostro_mp(rostro, self.rostros_conocidos)
                
                if nombre is None or nombre.strip() == "":
                    nombre = "Desconocido"
                    
                # Color del marco
                color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)

                # Evita duplicados
                if nombre not in self.rostros_detectados:
                    self.rostros_detectados.add(nombre)
                    fecha, hora = get_fecha_hora()
                    registrar_asistencia(nombre, fecha, hora)
                    if nombre == "Desconocido":
                        guardar_rostro_desconocido(rostro, fecha, hora)

                # Dibuja el recuadro
                cv2.rectangle(frame, (x, y), (x+ancho, y+alto), color, 2)
                cv2.putText(frame, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Fecha y hora
        fecha, hora = get_fecha_hora()
        cv2.putText(frame, f"{fecha} {hora}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Mostrar en ventana Tkinter
        imagen_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video_label.imgtk = imagen_tk
        self.video_label.configure(image=imagen_tk)

        self.root.after(10, self.mostrar_video)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# Ejecutar
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
