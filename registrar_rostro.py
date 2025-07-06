import cv2
import os

def registrar_rostro(nombre, cantidad=3):
    carpeta = "rostros_conocidos"
    os.makedirs(carpeta, exist_ok=True)

    cap = cv2.VideoCapture(0)
    contador = 0

    print(f"[INFO] Capturando rostros para: {nombre}...")
    print("Presiona 'c' para capturar, 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mostrar vista previa
        cv2.imshow("Registrar Rostro", frame)
        key = cv2.waitKey(1) & 0xFF

        # Capturar rostro
        if key == ord('c') and contador < cantidad:
            rostro = frame.copy()
            nombre_archivo = f"{nombre}_{contador+1}.jpg"
            cv2.imwrite(os.path.join(carpeta, nombre_archivo), rostro)
            print(f"[✔] Imagen {contador+1} guardada.")
            contador += 1

        # Salir manualmente
        if key == ord('q') or contador == cantidad:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[FINALIZADO] Se registraron {contador} imágenes para '{nombre}'.")

# Uso directo desde terminal
if __name__ == "__main__":
    nombre = input("Ingrese el nombre de la persona a registrar: ").strip()
    registrar_rostro(nombre, cantidad=3)
