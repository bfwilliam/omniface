import tkinter as tk
from tkinter import messagebox, ttk
import os
import cv2

CARPETA = "rostros_conocidos"
os.makedirs(CARPETA, exist_ok=True)

def capturar_rostros(nombre, cantidad=3):
    cap = cv2.VideoCapture(0)
    contador = 0
    messagebox.showinfo("Captura", f"Presiona 'c' para capturar {cantidad} imágenes. 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Captura de Rostros", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and contador < cantidad:
            archivo = f"{nombre}_{contador+1}.jpg"
            path = os.path.join(CARPETA, archivo)
            cv2.imwrite(path, frame)
            contador += 1
            print(f"[✔] Imagen {contador} guardada.")

        if key == ord('q') or contador == cantidad:
            break

    cap.release()
    cv2.destroyAllWindows()
    actualizar_historial()

def registrar():
    nombre = entry_nombre.get().strip()
    if not nombre:
        messagebox.showwarning("Nombre requerido", "Ingrese un nombre para registrar.")
        return
    capturar_rostros(nombre)

def actualizar_historial():
    historial = {}
    for archivo in os.listdir(CARPETA):
        if archivo.endswith(".jpg"):
            nombre = "_".join(archivo.split("_")[:-1])
            historial[nombre] = historial.get(nombre, 0) + 1

    historial_listbox.delete(0, tk.END)
    for nombre, total in historial.items():
        historial_listbox.insert(tk.END, f"{nombre} ({total} imágenes)")

def eliminar_usuario():
    seleccion = historial_listbox.curselection()
    if not seleccion:
        messagebox.showwarning("Seleccionar usuario", "Seleccione un nombre del historial.")
        return

    texto = historial_listbox.get(seleccion[0])
    nombre = texto.split(" (")[0]

    confirmar = messagebox.askyesno("Confirmar eliminación", f"¿Deseas eliminar todas las imágenes de '{nombre}'?")
    if not confirmar:
        return

    # Eliminar archivos
    archivos = os.listdir(CARPETA)
    eliminados = 0
    for archivo in archivos:
        if archivo.startswith(nombre + "_") and archivo.endswith(".jpg"):
            os.remove(os.path.join(CARPETA, archivo))
            eliminados += 1

    messagebox.showinfo("Eliminación exitosa", f"Se eliminaron {eliminados} imágenes de '{nombre}'.")
    actualizar_historial()

# Interfaz gráfica
ventana = tk.Tk()
ventana.title("Registro de Rostros")
ventana.geometry("400x400")
ventana.resizable(False, False)

# Entrada
label = tk.Label(ventana, text="Nombre del estudiante:")
label.pack(pady=10)
entry_nombre = tk.Entry(ventana, width=30)
entry_nombre.pack()

# Botones
btn = tk.Button(ventana, text="Registrar rostro", command=registrar)
btn.pack(pady=10)
btn_eliminar = tk.Button(ventana, text="Eliminar seleccionado", command=eliminar_usuario)
btn_eliminar.pack(pady=5)

# Historial
tk.Label(ventana, text="Historial de registros:").pack(pady=5)
historial_listbox = tk.Listbox(ventana, width=40, height=10)
historial_listbox.pack(pady=5)

# Cargar historial
actualizar_historial()

ventana.mainloop()
