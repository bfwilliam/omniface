import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pickle
import mediapipe as mp
from datetime import datetime

# Rutas y módulos personalizados
PROYECTO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROYECTO_RAIZ not in sys.path:
    sys.path.insert(0, PROYECTO_RAIZ)

from utils.helpers import get_fecha_hora, registrar_asistencia, verificar_duplicado
from deepface import DeepFace

# Cargar embeddings o inicializar vacíos
EMBEDDINGS_PATH = "reconocedor/embeddings.pkl"
if os.path.exists(EMBEDDINGS_PATH):
    try:
        with open(EMBEDDINGS_PATH, "rb") as f:
            base_embeddings = pickle.load(f)
            print(f"[📦] {len(base_embeddings)} usuario(s) cargado(s) desde embeddings.pkl")
    except Exception as e:
        print(f"[❌] Error al cargar embeddings: {e}")
        base_embeddings = {}
else:
    print("[⚠️] No se encontró el archivo embeddings.pkl, se inicializa vacío.")
    base_embeddings = {}

# Umbral de distancia para considerar un rostro como conocido
UMBRAL_DISTANCIA = 10

def reconocer_rostro_con_facemesh(frame, bbox):
    x, y, w, h = bbox
    rostro = frame[y:y+h, x:x+w]

    try:
        representacion = DeepFace.represent(img_path=rostro, model_name="Facenet", enforce_detection=False)
        if not representacion or "embedding" not in representacion[0]:
            return "Desconocido", (x, y, w, h)

        embedding_actual = np.array(representacion[0]["embedding"])
        nombre_reconocido = "Desconocido"
        distancia_min = float("inf")

        for nombre, embedding_registrado in base_embeddings.items():
            distancia = np.linalg.norm(embedding_actual - embedding_registrado)
            if distancia < distancia_min:
                distancia_min = distancia
                nombre_reconocido = nombre

        if distancia_min < UMBRAL_DISTANCIA:
            return nombre_reconocido.capitalize(), (x, y, w, h)
        else:
            return "Desconocido", (x, y, w, h)
    except Exception as e:
        print(f"[!] Error en reconocimiento: {e}")
        return "Desconocido", (x, y, w, h)

class App:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Sistema de Control de Presencia")

        # Captura de video
        self.video = cv2.VideoCapture(0)

        # Interfaz
        self.label_video = tk.Label(ventana)
        self.label_video.pack()

        self.entry_nombre = tk.Entry(self.ventana, width=30)
        self.entry_nombre.pack(pady=5)
        self.entry_nombre.insert(0, "Nombre del nuevo usuario")

        self.btn_capturar = tk.Button(self.ventana, text="📸 Capturar Rostro", command=self.capturar_rostro)
        self.btn_capturar.pack(pady=5)

        self.mostrar_video()

    def mostrar_video(self):
        ret, frame = self.video.read()
        if not ret:
            self.ventana.after(10, self.mostrar_video)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        altura, ancho, _ = frame.shape
        resultados = []

        with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as detector:
            salida = detector.process(frame_rgb)
            if salida.detections:
                resultados = salida.detections

        for detection in resultados:
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * ancho)
            y = int(bboxC.ymin * altura)
            w = int(bboxC.width * ancho)
            h = int(bboxC.height * altura)

            nombre, (x, y, w, h) = reconocer_rostro_con_facemesh(frame, (x, y, w, h))
            color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            fecha, hora = get_fecha_hora()
            cv2.putText(frame, f"{fecha} {hora}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if nombre != "Desconocido" and not verificar_duplicado(nombre, fecha, hora):
                registrar_asistencia(nombre, fecha, hora)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.label_video.imgtk = imgtk
        self.label_video.configure(image=imgtk)
        self.ventana.after(10, self.mostrar_video)

    def capturar_rostro(self):
        nombre = self.entry_nombre.get().strip().lower()
        if not nombre:
            messagebox.showwarning("Advertencia", "Ingrese un nombre válido.")
            return

        carpeta = "rostros_conocidos"
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)

        capturas = []
        for i in range(3):
            ret, frame = self.video.read()
            if not ret:
                messagebox.showerror("Error", f"No se pudo capturar la imagen {i + 1}.")
                return
            ruta = os.path.join(carpeta, f"{nombre}_{i+1}.jpg")
            cv2.imwrite(ruta, frame)
            capturas.append(ruta)
            print(f"[📸] Imagen {i+1} guardada: {ruta}")
            cv2.waitKey(300)

        messagebox.showinfo("Captura completa", f"Se guardaron 3 fotos de {nombre}.")
        self.generar_y_actualizar_embeddings(nombre, capturas)

    def generar_y_actualizar_embeddings(self, nombre, capturas):
        try:
            embeddings = []
            print(f"\n[🧠] Generando embeddings para {nombre}...")
            for ruta in capturas:
                representacion = DeepFace.represent(img_path=ruta, model_name="Facenet", enforce_detection=False)
                if representacion and isinstance(representacion, list):
                    embedding = representacion[0].get("embedding")
                    if embedding:
                        embeddings.append(np.array(embedding))
                        print(f"[✔] Embedding generado desde {ruta}")
                    else:
                        print(f"[!] Embedding vacío en {ruta}")
                else:
                    print(f"[!] No se detectó rostro en {ruta}")

            if not embeddings:
                messagebox.showerror("Error", "No se pudo generar ningún embedding. Intente nuevamente.")
                return

            embedding_promedio = np.mean(embeddings, axis=0)
            base_embeddings[nombre] = embedding_promedio

            with open(EMBEDDINGS_PATH, "wb") as f:
                pickle.dump(base_embeddings, f)

            print(f"[💾] Embedding de {nombre} actualizado en {EMBEDDINGS_PATH}")
            messagebox.showinfo("Éxito", "Embeddings actualizados correctamente.")

        except Exception as e:
            print(f"[❌] Error al generar embeddings: {e}")
            messagebox.showerror("Error", f"No se pudo generar el embedding:\n{e}")

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

if __name__ == "__main__":
    ventana = tk.Tk()
    app = App(ventana)
    ventana.mainloop()
