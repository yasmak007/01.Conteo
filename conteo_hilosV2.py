import cv2
import numpy as np

# --- Cargar imagen ---
img = cv2.imread('Cinta-Hilos.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Top-Hat para realzar líneas claras sobre fondo oscuro
kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)

# --- AUMENTAR CONTRASTE ---
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)

# --- SUAVIZAR Y AUMENTAR NITIDEZ ---
blur_for_sharp = cv2.GaussianBlur(enhanced, (5,5), 0)
sharp = cv2.addWeighted(enhanced, 1.5, blur_for_sharp, -0.005, 0)

# --- UMBRAL ADAPTATIVO ---
th = cv2.adaptiveThreshold(
    sharp, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31, 5
)

# --- MORFOLOGÍA ---
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

# --- Encontrar contornos ---
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- Filtrar hilos por área ---
hilos_filtrados = []
medidas_filtradas = []
areas_filtradas = []

AREA_MIN = 10000
AREA_MAX = 30000

for idx, c in enumerate(contours):
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)  # opcional, para anotación de texto

    if AREA_MIN <= area <= AREA_MAX:
        hilos_filtrados.append(c)
        medidas_filtradas.append((w, h))
        areas_filtradas.append(area)

        # Dibujar contorno real del hilo
        cv2.drawContours(img, [c], -1, (0,0,255), 2)
        # Poner número
        cv2.putText(img, f"{len(hilos_filtrados)}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# --- Resultados ---
print("Número de hilos detectados en rango de área:", len(hilos_filtrados))


# Mostrar imagen
img_resized = cv2.resize(img, (800, 800))
cv2.imshow('Hilos filtrados por área (contorno real)', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

