import cv2
import numpy as np

def aplicar_gamma(imagen, gamma=0.8):
    """
    Aplica la corrección Gamma. 
    gamma < 1.0 hace las áreas oscuras más brillantes (aumenta su intensidad).
    """
    # Crear una tabla de mapeo de píxeles
    invGamma = 1.0 / gamma
    # Se utiliza una tabla de consulta (LUT) para aplicar la transformación rápidamente
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    # Aplicar la corrección Gamma a la imagen
    return cv2.LUT(imagen, table)


def contar_filamentos_gamma(ruta_imagen):
    # --- Cargar imagen ---
    img = cv2.imread(ruta_imagen)
    if img is None:
        print(f"Error: No se pudo cargar la imagen '{ruta_imagen}'. Revise la ruta.")
        return 0
    
    img_procesada = img.copy()
    gray = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2GRAY)

    # --- AUMENTAR CONTRASTE (CLAHE) ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # --- 💡 1. TRANSFORMACIÓN GAMMA (Realza sombras/oscuros) ---
    # Usamos gamma = 0.8 para aumentar la intensidad de las zonas oscuras (sombras)
    gamma_corrected = aplicar_gamma(enhanced, gamma=0.8)
    
    # --- SUAVIZAR Y AUMENTAR NITIDEZ ---
    # Usamos la imagen con Gamma
    blur_for_sharp = cv2.GaussianBlur(gamma_corrected, (5,5), 0)
    # Acentuación suave
    sharp = cv2.addWeighted(gamma_corrected, 1.5, blur_for_sharp, -0.005, 0)
    
    # --- UMBRAL ADAPTATIVO (Invertido) ---
    # Usamos THRESH_BINARY_INV, asumiendo que los *canales* (sombras) son oscuros y queremos contarlos.
    # El Laplaciano suele funcionar mejor con esta imagen, pero adaptaremos tu lógica.
    th = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, # Invertido para que los canales oscuros sean blancos
        31, 5
    )

    # --- MORFOLOGÍA: Limpieza y Clausura (Ajustada para prevenir uniones) ---
    # Usamos el kernel original (7, 3) pero con menos iteraciones para evitar unir
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    
    # Una sola iteración de clausura
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Apertura para limpiar ruido
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- Encontrar contornos (Canales/Sombras) ---
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # -------------------------------------------------------------
    # --- FILTRO FINAL: Por Ancho (w) (Para Contar Hilos = Canales + 1) ---
    # -------------------------------------------------------------
    canales_filtrados = []

    # Parámetros de filtro por ancho (más robusto que el área)
    W_MIN = 10  # Ancho mínimo del canal/sombra
    W_MAX = 70  # Ancho máximo (si es más ancho, probablemente dos se han fusionado)
    H_MIN = img_procesada.shape[0] * 0.2 # Mínimo 20% de la altura total
    
    for idx, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c) 
        area = cv2.contourArea(c)
        
        # Filtramos por altura y ancho para aislar solo los canales verticales.
        if (W_MIN <= w <= W_MAX) and (h > H_MIN) and (area > 500):
            canales_filtrados.append(c)
            
            # Dibujar el contorno del CANAL detectado
            cv2.drawContours(img_procesada, [c], -1, (0,0,255), 2)
            
            # Poner número del CANAL
            cv2.putText(img_procesada, 
                        f"C{len(canales_filtrados)}", 
                        (x, y + h // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), # Verde
                        2)

    # El número de FILAMENTOS es igual al número de CANALES (picos) + 1
    num_canales = len(canales_filtrados)
    num_filamentos = num_canales + 1 

    # --- Resultados ---
    print("\n" + "="*50)
    print(f"Número de CANALES (picos/sombras) detectados: {num_canales}")
    print(f"✅ RESULTADO FINAL (Filamentos = Canales + 1): **{num_filamentos}**")
    print("="*50)

    # Mostrar imagen
    alto_total, ancho_total, _ = img.shape
    if alto_total > 400 or ancho_total > 400:
        img_resized = cv2.resize(img_procesada, (600, int(alto_total * (400 / ancho_total))))
    else:
        img_resized = img_procesada
        
    cv2.imshow('Hilos Contados (Canales Enumerados)', img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return num_filamentos

# --- Ejecución ---
ruta_del_archivo = "Cinta-Hilos.jpg" 
conteo_final = contar_filamentos_gamma(ruta_del_archivo)