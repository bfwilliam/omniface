import cv2

cap = cv2.VideoCapture(1)  # Media Foundation

if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se puede recibir el frame")
        break

    cv2.imshow("Video Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
