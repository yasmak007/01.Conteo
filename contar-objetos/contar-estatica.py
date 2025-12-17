import cv2
import numpy as np

print('Libreria cv2 cargada correctamente')

# --- Colores ---
color_texto = (255, 255, 0)
color_numero = (0, 255, 255)
color_cierre = 255
color_canny_rojo = (0, 0, 255)
color_linea_recta = (0, 255, 0)  # Verde para líneas rectas

# --- Parámetros ---
min_altura = 50  # Altura mínima para considerar línea vertical

# --- Cargar imagen ---
img_ruta = r"c:\Users\yassine.makhloufi\OneDrive - AITEX\Escritorio\VScode\01.Conteo\contar-objetos\Cinta.jpg"
img = cv2.imread(img_ruta)
if img is None:
    raise FileNotFoundError("No se pudo cargar la imagen. Revisa la ruta.")

img = cv2.resize(img, (int(2643 / 3), int(1750 / 2)))

# --- Imagen de resta ---
img_negativa = cv2.bitwise_not(img)
img_resta = cv2.subtract(img, img_negativa)
gray_resta = cv2.cvtColor(img_resta, cv2.COLOR_BGR2GRAY)

# --- Línea de cierre inferior ---
h, w = gray_resta.shape
cv2.line(gray_resta, (0, h-1), (w, h-1), color_cierre, 5)

# --- Erosión + dilatación ---
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
gray_eroded = cv2.erode(gray_resta, kernel, iterations=1)
gray_dilated = cv2.dilate(gray_eroded, kernel, iterations=2)

# --- Canny ---
canny = cv2.Canny(gray_dilated, 300, 100)
mask_canny = canny > 0
img[mask_canny] = color_canny_rojo

# --- Contornos ---
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar líneas verticales
lineas_verticales = []
for cnt in contours:
    x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
    if h_cnt >= min_altura and h_cnt > 2*w_cnt:
        lineas_verticales.append((x, y, w_cnt, h_cnt))

# Ordenar por y
lineas_verticales = sorted(lineas_verticales, key=lambda c: c[1])

# --- Aproximar a líneas rectas mediante media de x ---
if lineas_verticales:
    # Extraer todas las x para hacer una media
    xs = [x + w//2 for x, y, w, h in lineas_verticales]
    x_media = int(np.mean(xs))

    # Dibujar líneas rectas verticales
    for x, y, w, h in lineas_verticales:
        cv2.line(img, (x_media, y), (x_media, y+h), color_linea_recta, 2)

# Texto resumen
texto = f'Líneas verticales detectadas: {len(lineas_verticales)}'
cv2.putText(img, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_texto, 2)
print(texto)

# Mostrar resultados
cv2.imshow('Líneas rectas promediadas', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
