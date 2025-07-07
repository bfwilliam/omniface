import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import mediapipe as mp

# Rutas y módulos personalizados
PROYECTO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROYECTO_RAIZ not in sys.path:
    sys.path.insert(0, PROYECTO_RAIZ)

from detectores.detector_mediapipe import detectar_rostros_mediapipe
from utils.helpers import get_fecha_hora, registrar_asistencia, verificar_duplicado
from reconocedor.reconocedor_mp import reconocer_rostro_mp
from utils.gestor_embeddings import generar_embedding_individual, actualizar_embeddings

class App:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Sistema de Control de Presencia")

        self.video = cv2.VideoCapture(0)

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
        detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
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
            cv2.waitKey(300)

        messagebox.showinfo("Captura completa", f"Se guardaron 3 fotos de {nombre}.")

        # Generar y actualizar embeddings usando el nuevo módulo
        embedding = generar_embedding_individual(nombre, capturas)
        if embedding is not None:
            exito = actualizar_embeddings(nombre, embedding)
            if exito:
                messagebox.showinfo("Éxito", "Embeddings actualizados correctamente.")
            else:
                messagebox.showerror("Error", "No se pudo actualizar el archivo de embeddings.")
        else:
            messagebox.showerror("Error", "No se generaron embeddings válidos.")

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

if __name__ == "__main__":
    ventana = tk.Tk()
    app = App(ventana)
    ventana.mainloop()
