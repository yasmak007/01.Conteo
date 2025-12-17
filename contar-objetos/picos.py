import cv2
import numpy as np

def procesar_cable(frame):
    # 1. Preprocesamiento: Convertir a gris y desenfocar ligeramente
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Detección de bordes (Canny) para resaltar los hilos
    # Ajusta los umbrales 50 y 150 según tu iluminación
    edges = cv2.Canny(blurred, 50, 150)

    # 3. Dilatación para conectar líneas rotas
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # 4. Encontrar contornos (cada hilo es un contorno alargado)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos por área y forma (solo verticales)
    pines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 50 and w < 20:  # Ajustar según tamaño del cable en pantalla
            pines.append((x, y, w, h))

    # Ordenar pines de izquierda a derecha
    pines.sort(key=lambda x: x[0])

    # 5. Dibujar y numerar
    for i, (x, y, w, h) in enumerate(pines):
        # Dibujar rectángulo alrededor del hilo
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Escribir el número del pin
        cv2.putText(frame, str(i + 1), (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

# --- Ejecución en Tiempo Real ---
# Cambia '0' por la ruta de un video o imagen si no tienes cámara conectada
cap = cv2.VideoCapture(0)

print("Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resultado = procesar_cable(frame)
    
    cv2.imshow('Detector de Pines para Corte', resultado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()