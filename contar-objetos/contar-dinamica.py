import cv2
import numpy as np

print('libreria cv2 cargada correctamente')

# --- Parámetros ---
borde = (0, 0, 255)
color_texto = (255, 255, 0)
color_punto = (0, 255, 0)

area_min = 500  # filtrar ruido
objeto_seleccionado = 1  # por defecto

# --- Cámara ---
cap = cv2.VideoCapture(0)  # 0 = webcam principal

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (450, 450))

    # --- Imagen de resta (igual que tu lógica) ---
    img_negativa = cv2.bitwise_not(frame)
    img_resta = cv2.subtract(frame, img_negativa)
    gray_resta = cv2.cvtColor(img_resta, cv2.COLOR_BGR2GRAY)

    # --- Canny ---
    canny = cv2.Canny(gray_resta, 300, 105)

    contours, _ = cv2.findContours(
        canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # --- Filtrar contornos pequeños ---
    objetos = []
    for cnt in contours:
        if cv2.contourArea(cnt) > area_min:
            objetos.append(cnt)

    # --- Ordenar por eje X (cinta horizontal) ---
    objetos = sorted(objetos, key=lambda c: cv2.boundingRect(c)[0])

    # --- Dibujar todos los contornos ---
    cv2.drawContours(frame, objetos, -1, borde, 2)

    # --- Seleccionar objeto ---
    if 0 < objeto_seleccionado <= len(objetos):
        cnt = objetos[objeto_seleccionado - 1]

        # Centro del objeto
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 6, color_punto, -1)
            cv2.putText(
                frame,
                f"Objeto {objeto_seleccionado} -> ({cx},{cy})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_texto,
                2
            )

    # --- Texto general ---
    cv2.putText(
        frame,
        f"Objetos detectados: {len(objetos)}",
        (10, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color_texto,
        2
    )

    cv2.imshow("Cinta - Deteccion", frame)
    cv2.imshow("Canny", canny)

    # --- Teclado ---
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    # Teclas 1–9 para seleccionar objeto
    if ord('1') <= key <= ord('9'):
        objeto_seleccionado = key - ord('0')

cap.release()
cv2.destroyAllWindows()
