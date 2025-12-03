import cv2
import numpy as np

def delimitar_y_enumerar_carriles(ruta_imagen):
    # --- Cargar la imagen ---
    img_original = cv2.imread(ruta_imagen)
    
    if img_original is None:
        print(f"Error: No se pudo cargar la imagen '{ruta_imagen}'. Revise la ruta.")
        return 0

    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    alto, ancho, _ = img_original.shape
    
    # --- 1. Generar la Imagen de Resta (Alto Contraste) ---
    img_negativa = cv2.bitwise_not(img_original)
    img_resta = cv2.subtract(img_original, img_negativa)
    
    # Convertir a escala de grises para la binarización
    gray_resta = cv2.cvtColor(img_resta, cv2.COLOR_BGR2GRAY)

    # --- 2. Binarización ADAPTATIVA (Clave para detectar objetos) ---
    # Usamos THRESH_BINARY ya que los hilos brillantes en la resta deberían ser blancos.
    
    # Bloque de 31 y C=5 es un buen punto de partida.
    img_binaria = cv2.adaptiveThreshold(
        gray_resta, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        31, # Tamaño del bloque de píxeles
        5   # Constante C: debe ser ajustada si la imagen binaria es negra.
    )
    
    # --- 3. Morfología para limpiar ruido y unir segmentos ---
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) 
    opened = cv2.morphologyEx(img_binaria, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Usamos una clausura vertical para unir segmentos rotos de los hilos.
    kernel_close_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20)) 
    final_binary = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close_vertical, iterations=1)

    # --- 4. Detección de Contornos ---
    contours, _ = cv2.findContours(final_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 5. Filtrado, Delimitación y Enumeración de los Carriles ---
    carriles_detectados = []
    
    # Parámetros de filtro (MUY tolerantes para asegurar detección)
    W_MAX = 70      # Ancho máximo 
    H_MIN = alto * 0.02 # Mínimo 2% de la altura de la imagen
    AREA_MIN = 100  # Área mínima muy baja

    for c in contours:
        x, y, w, h = cv2.boundingRect(c) 
        area = cv2.contourArea(c)
        
        # Criterio: Debe ser alto (H_MIN), no demasiado ancho (W_MAX) y tener área (AREA_MIN).
        es_carril = (h > H_MIN) and (w <= W_MAX) and (area > AREA_MIN)
        
        if es_carril:
            carriles_detectados.append(c)
            
            # Enmarcación: Dibujar un rectángulo delimitador (Bounding Box)
            cv2.rectangle(img_resta, (x, y), (x + w, y + h), (0, 255, 255), 2) # Amarillo
            
            # Enumeración del carril
            cv2.putText(img_resta, 
                        str(len(carriles_detectados)), 
                        (x + w // 2 - 10, y + h // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 0, 255), # Rojo
                        2)

    # --- 6. Mostrar Resultado ---
    num_detectados = len(carriles_detectados)
    print(f"\n✅ Número de carriles (filamentos) detectados y enumerados: **{num_detectados}**")
    
    display_width = 600
    display_height = int(alto * (display_width / ancho))

    img_final_disp = cv2.resize(img_resta, (display_width, display_height))
    img_binary_disp = cv2.resize(final_binary, (display_width, display_height))

    # Muestra la imagen binaria para que veas qué está detectando (CRÍTICO)
    cv2.imshow('1. IMAGEN BINARIA (DEPURACION)', img_binary_disp) 
    cv2.imshow('2. RESULTADO FINAL: Carriles Enumerados', img_final_disp)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return num_detectados

# --- Ejecución ---
ruta_del_archivo = "Cinta-Hilos.jpg"
conteo_final = delimitar_y_enumerar_carriles(ruta_del_archivo)