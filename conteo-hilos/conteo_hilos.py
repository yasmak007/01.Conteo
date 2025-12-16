import cv2
import numpy as np

# --- Cargar imagen ---
img = cv2.imread('Cinta-Hilos.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Top-Hat para realzar líneas claras sobre fondo oscuro
kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)

# --- AUMENTAR CONTRASTE (muy importante) ---
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)

# --- SUAVIZAR PARA QUITAR RUIDO ---
blur = cv2.GaussianBlur(enhanced, (5,5), 0)

# --- UMBRAL ADAPTATIVO (mucho mejor que threshold fijo) ---
th = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31, 5
)

# --- MORFOLOGÍA: unir trozos de los hilos ---
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

# --- Limpiar ruido pequeño ---
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

# --- Encontrar contornos reales ---
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar por tamaño: evitar manchas pequeñas
hilos = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    # Filtro inteligente: los hilos son largos y finos
    if w*h > 500 and max(w,h) > 40:  
        hilos.append(c)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)

# --- Resultado ---
print("Número de hilos detectados:", len(hilos))

# Mostrar (opcional)
img_resized = cv2.resize(img, (800, 800))
cv2.imshow('Hilos detectados', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
