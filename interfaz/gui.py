import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np

# Agregar ruta raíz del proyecto
PROYECTO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROYECTO_RAIZ not in sys.path:
    sys.path.insert(0, PROYECTO_RAIZ)

from detectores.detector_mediapipe import detectar_rostros_mediapipe
from utils.helpers import get_fecha_hora, registrar_asistencia, verificar_duplicado
from reconocedor.reconocedor_mp import reconocer_rostro_mp
from reconocedor.entrenar_embeds import generar_embeddings  # <- Entrenamiento centralizado

class App:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Sistema de Control de Presencia")

        self.video = cv2.VideoCapture(0)

        # Interfaz de usuario
        self.label_video = tk.Label(ventana)
        self.label_video.pack()

        self.entry_nombre = tk.Entry(ventana, width=30)
        self.entry_nombre.pack(pady=5)
        self.entry_nombre.insert(0, "Nombre del nuevo usuario")

        self.btn_capturar = tk.Button(ventana, text="📸 Capturar Rostro", command=self.capturar_rostro)
        self.btn_capturar.pack(pady=5)

        self.mostrar_video()

    def mostrar_video(self):
        ret, frame = self.video.read()
        if not ret:
            self.ventana.after(10, self.mostrar_video)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.8)
        resultados = detector.process(frame_rgb)

        if resultados.detections:
            for detection in resultados.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                nombre_bbox = reconocer_rostro_mp(frame, (x, y, w, h))
                if nombre_bbox is None:
                    continue

                nombre, (x, y, w, h) = nombre_bbox
                color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                fecha, hora = get_fecha_hora()
                cv2.putText(frame, f"{fecha} {hora}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if not verificar_duplicado(nombre, fecha, hora):
                    registrar_asistencia(nombre, fecha, hora)

        # Mostrar en GUI
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
        os.makedirs(carpeta, exist_ok=True)

        mp_face_detection = mp.solutions.face_detection
        detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

        capturas = []
        intentos = 0
        max_intentos = 7

        print(f"\n[📸] Iniciando captura de imágenes para '{nombre}'...")

        while len(capturas) < 3 and intentos < max_intentos:
            ret, frame = self.video.read()
            intentos += 1

            if not ret:
                print(f"[✖] Intento {intentos}: error al leer frame.")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = detector.process(frame_rgb)

            if resultado.detections:
                ruta = os.path.join(carpeta, f"{nombre}_{len(capturas)+1}.jpg")
                cv2.imwrite(ruta, frame)
                capturas.append(ruta)
                print(f"[✔] Imagen {len(capturas)} guardada: {ruta}")
            else:
                print(f"[!] Intento {intentos}: rostro no detectado.")

            cv2.waitKey(300)

        if len(capturas) < 3:
            messagebox.showerror("Error", f"No se capturaron suficientes imágenes válidas. ({len(capturas)} de 3 requeridas)")
            return

        messagebox.showinfo("Captura completa", f"Se guardaron {len(capturas)} fotos de {nombre}.")

        # --- Embeddings ---
        try:
            from deepface import DeepFace
            import numpy as np
            import pickle

            print(f"\n[🧠] Generando embeddings para {nombre}...")
            embeddings = []

            for ruta in capturas:
                representacion = DeepFace.represent(img_path=ruta, model_name="Facenet", enforce_detection=False)
                if representacion and isinstance(representacion, list):
                    embedding = representacion[0].get("embedding")
                    if embedding:
                        embeddings.append(np.array(embedding))
                        print(f"   - {os.path.basename(ruta)} [✔]")
                    else:
                        print(f"   - {os.path.basename(ruta)} [✖] Embedding vacío")
                else:
                    print(f"   - {os.path.basename(ruta)} [✖] Representación inválida")

            if not embeddings:
                messagebox.showerror("Error", "No se pudo generar ningún embedding. Intente nuevamente.")
                return

            embedding_promedio = np.mean(embeddings, axis=0)

            # Guardar en embeddings.pkl
            embeddings_path = "reconocedor/embeddings.pkl"
            if os.path.exists(embeddings_path):
                with open(embeddings_path, "rb") as f:
                    data = pickle.load(f)
            else:
                data = {}

            data[nombre] = embedding_promedio

            with open(embeddings_path, "wb") as f:
                pickle.dump(data, f)

            print(f"[💾] Embedding de {nombre} actualizado en embeddings.pkl\n")
            messagebox.showinfo("Éxito", "Embeddings actualizados correctamente.")

        except Exception as e:
            print(f"[❌] Error al generar embedding para {nombre}: {e}")
            messagebox.showerror("Error", f"No se pudo generar el embedding:\n{e}")


    def __del__(self):
        if self.video.isOpened():
            self.video.release()

if __name__ == "__main__":
    ventana = tk.Tk()
    app = App(ventana)
    ventana.mainloop()
