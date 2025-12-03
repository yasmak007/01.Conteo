import cv2
import numpy as np

def detectar_21_lineas_hough(ruta_imagen):
    # --- Cargar la imagen ---
    img_original = cv2.imread(ruta_imagen)
    
    if img_original is None:
        print(f"Error: No se pudo cargar la imagen '{ruta_imagen}'. Revise la ruta.")
        return 0

    alto, ancho, _ = img_original.shape
    
    # 🚨 CAMBIO 1: Usar la imagen en escala de grises directamente para realzar los surcos/sombras
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    # --- 2. Binarización ADAPTATIVA (CLAVE) ---
    # 🚨 CAMBIO 2: Usamos THRESH_BINARY_INV. Esto hace que los surcos oscuros (los límites) sean BLANCOS
    img_binaria = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, # <--- INVERTIMOS LA BINARIZACIÓN
        31, 
        2   
    )
    
    # --- 3. OPERACIONES MORFOLÓGICAS Y CONVOLUCIÓN ---
    # Usamos una Clausura más fuerte (Dilatación seguida de Erosión) para unir los surcos blancos
    
    # Eliminamos la Erosión inicial ya que los surcos son ya delgados.
    
    # Dilatación (Une los segmentos de los surcos/líneas)
    kernel_dilatacion = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_dilatada = cv2.dilate(img_binaria, kernel_dilatacion, iterations=1)
    
    # Apertura para limpiar ruido pequeño (Erosión seguida de Dilatación)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    final_binary = cv2.morphologyEx(img_dilatada, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # --- 4. Detección de Líneas usando cv2.HoughLinesP ---
    
    # Longitud mínima: Bajamos al 20% para capturar líneas que no van de punta a punta
    H_MIN_PIXELS = int(alto * 0.20) 

    lines = cv2.HoughLinesP(
        final_binary, 
        rho=1,
        theta=np.pi/180,
        threshold=25, # Umbral muy bajo para forzar la detección de 21 líneas
        minLineLength=H_MIN_PIXELS,
        maxLineGap=15 # Aumentamos un poco el Gap para que los segmentos rotos se unan
    )

    # --- 5. Filtrado de Líneas Verticales y Medición ---
    lineas_detectadas = []
    
    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            
            # 🚨 CAMBIO 3: Filtro de verticalidad más flexible.
            # Permitimos hasta 20 píxeles de inclinación (antes era 10)
            if abs(x2 - x1) < 20: 
                
                # Medidas:
                anchura_linea = abs(x2 - x1) 
                altura_linea = abs(y2 - y1) 
                x_pos_centro = (x1 + x2) // 2
                
                lineas_detectadas.append({
                    'id': len(lineas_detectadas) + 1,
                    'x': x_pos_centro, 
                    'y_start': y1, 
                    'y_end': y2, 
                    'w': anchura_linea, 
                    'h': altura_linea,
                    'coords': line[0]
                })

    # Ordenar las líneas de izquierda a derecha y re-enumerar
    lineas_detectadas.sort(key=lambda item: item['x'])
    for i, linea in enumerate(lineas_detectadas):
        linea['id'] = i + 1

    num_filtrados = len(lineas_detectadas)
    
    # --- 6. Dibujar Enumeración y Enmarcación ---
    
    medidas_impresas = []
    # Usaremos una copia de la imagen original para dibujar sobre ella (más claro)
    img_dibujo = img_original.copy() 

    for linea in lineas_detectadas:
        idx = linea['id']
        x, y1, y2, h_line = linea['x'], linea['y_start'], linea['y_end'], linea['h']
        x1, _, x2, _ = linea['coords']
        
        min_y = min(y1, y2)
        max_y = max(y1, y2)

        # Enmarcación (Amarillo)
        cv2.rectangle(img_dibujo, (x1 - 5, min_y), (x2 + 5, max_y), (0, 255, 255), 2)
        
        # Enumeración de la línea (Rojo)
        cv2.putText(img_dibujo, 
                    str(idx), 
                    (x - 10, min_y + h_line // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2)
        
        medidas_impresas.append(f"Línea {idx}: Anchura={linea['w']} px, Altura={h_line} px, Centro X={x} px")

    # --- 7. Mostrar Resultado y Medidas ---
    
    print("\n" + "="*70)
    print(f"✅ Se detectaron **{num_filtrados}** líneas. (Objetivo: 21)")
    print("\n📐 Medidas de cada línea detectada (en píxeles):")
    if medidas_impresas:
        for medida in medidas_impresas:
            print(f"- {medida}")
        
    print("="*70)
    
    display_width = 600
    display_height = int(alto * (display_width / ancho))

    img_final_disp = cv2.resize(img_dibujo, (display_width, display_height))
    img_binary_disp = cv2.resize(final_binary, (display_width, display_height))

    cv2.imshow('1. IMAGEN BINARIA (SURCOS BLANCOS)', img_binary_disp) 
    cv2.imshow('2. RESULTADO FINAL: Lineas Contadas y Medidas', img_final_disp)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return num_filtrados

# --- Ejecución ---
ruta_del_archivo = "Cinta-Hilos.jpg"
conteo_final = detectar_21_lineas_hough(ruta_del_archivo)