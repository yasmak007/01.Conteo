import cv2
import numpy as np

def generar_resta_de_imagenes(ruta_imagen):
    # --- Cargar la imagen ---
    img_original = cv2.imread(ruta_imagen)
    
    if img_original is None:
        print(f"Error: No se pudo cargar la imagen '{ruta_imagen}'. Revise la ruta.")
        return

    # Obtener dimensiones
    alto, ancho, _ = img_original.shape
    
    # --- 1. Generar la Imagen Negativa ---
    # Invierte todos los bits (P' = 255 - P)
    img_negativa = cv2.bitwise_not(img_original)

    # --- 2. Resta de Imágenes (cv2.subtract) ---
    # Resta los valores de píxel: P_resta = P_original - P_negativa
    # Esto resalta las áreas donde la imagen original es más brillante que su negativo.
    img_resta = cv2.subtract(img_original, img_negativa)
    
    
    # --- 3. Mostrar Imágenes ---
    
    # Definir un tamaño de visualización consistente (ajustado a 600px de ancho)
    display_width = 600
    display_height = int(alto * (display_width / ancho))

    def resize_for_display(img):
        return cv2.resize(img, (display_width, display_height))

    # Redimensionar las imágenes
    img_original_disp = resize_for_display(img_original)
    img_resta_disp = resize_for_display(img_resta)

    # Mostrar resultados
    cv2.imshow('1. Imagen Original', img_original_disp)
    cv2.imshow('2. Resultado de la RESTA (Original - Negativo)', img_resta_disp)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# --- Ejecución ---
# Sustituye 'Cinta-Hilos.jpg' por la ruta de tu imagen si es diferente
ruta_del_archivo = "Cinta-Hilos.jpg"
generar_resta_de_imagenes(ruta_del_archivo)