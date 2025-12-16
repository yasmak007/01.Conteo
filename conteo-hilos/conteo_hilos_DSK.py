import cv2
import numpy as np
import matplotlib.pyplot as plt

def procesar_cinta(ruta_imagen):
    """
    Carga una imagen, separa la cinta blanca del fondo, y cuenta los filamentos 
    utilizando un método de proyección vertical y análisis de componentes conectados.

    :param ruta_imagen: La ruta al archivo de imagen.
    :return: El número estimado de filamentos.
    """
    try:
        # 1. Cargar la imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"Error: No se pudo cargar la imagen en la ruta: {ruta_imagen}")
            return None

        # --- NUEVA LÍNEA: Redimensionar a 450x450 píxeles ---
        # Usamos INTER_AREA para reducir el tamaño, ya que es el método de interpolación recomendado.
        imagen = cv2.resize(imagen, (1050, 750), interpolation=cv2.INTER_AREA)
        print("✅ Imagen redimensionada a 450x450 píxeles.")
        # --- FIN NUEVA LÍNEA ---

        # 2. Preprocesamiento: Convertir a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # 3. Binarización (Separación de fondo y objeto)
        # Umbral fijo para aislar la cinta blanca (valores > 200)
        _, binaria = cv2.threshold(gris, 200, 255, cv2.THRESH_BINARY)
        
        # Opcional: Operación de Apertura para reducir ruido fino
        kernel_ruido = np.ones((3, 3), np.uint8)
        abertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_ruido, iterations=1)

        # 4. Aislamiento de la Región de Interés (ROI)
        contornos, _ = cv2.findContours(abertura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Encontrar el contorno más grande (la cinta) y recortar la imagen a esa área
        contorno_cinta = max(contornos, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contorno_cinta)
        cinta_roi = gris[y:y+h, x:x+w]
        
        # 5. Conteo de filamentos usando Proyección Vertical
        
        # Sumar los píxeles verticalmente (a lo largo de las columnas)
        # Los picos corresponden a los filamentos
        perfil_vertical = np.sum(cinta_roi.astype(np.float32), axis=0) # Usar float para evitar desbordamiento
        
        # Normalizar el perfil y convertir a 8-bit para el procesamiento con OpenCV
        perfil_normalizado = perfil_vertical / perfil_vertical.max() * 255
        perfil_normalizado = perfil_normalizado.astype(np.uint8)

        # 6. Detección de picos
        # Se binariza el perfil para identificar las crestas de los filamentos.
        # El umbral (e.g., 180) debe ser lo suficientemente alto para capturar solo los picos.
        _, picos = cv2.threshold(perfil_normalizado, 180, 1, cv2.THRESH_BINARY)
        
        # Usamos una operación de Clausura para "juntar" puntos cercanos del mismo pico, 
        # asegurando que cada filamento se cuente como un solo componente.
        kernel_conteo = np.ones((1, 5), np.uint8) # Kernel horizontal
        picos_cerrados = cv2.morphologyEx(picos[np.newaxis, :], cv2.MORPH_CLOSE, kernel_conteo)
        
        # Contar el número de componentes conectados (filamentos)
        etiquetas, num_componentes = cv2.connectedComponents(picos_cerrados)
        
        # Restar 1 porque la etiqueta 0 es el fondo (el canal entre los filamentos)
        num_filamentos = num_componentes - 1

        # 7. Visualización (Para verificar la separación y el perfil)
        
        # Mostrar la imagen recortada para verificar la separación
        cv2.imshow("1. Cinta Aislada (ROI)", cinta_roi) 
        
        # Visualizar el perfil vertical y los picos detectados
        plt.figure(figsize=(10, 4))
        plt.plot(perfil_normalizado, label="Perfil de Intensidad Vertical")
        plt.plot(picos_cerrados[0] * perfil_normalizado.max(), label="Picos Detectados", linestyle='dashed')
        plt.title(f"Perfil Vertical de la Cinta (Conteo: {num_filamentos})")
        plt.xlabel("Columna de Píxeles")
        plt.ylabel("Suma de Intensidad Normalizada")
        plt.legend()
        plt.grid(axis='y')
        
        # Mueve la ventana de la gráfica si está superpuesta
        manager = plt.get_current_fig_manager()
        manager.window.wm_geometry("+0+500") # Mover a la posición x=0, y=500
        
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return num_filamentos

    except Exception as e:
        print(f"Ocurrió un error durante el procesamiento: {e}")
        return None

# --- Ejecución ---
# La ruta al archivo que has subido.
ruta_del_archivo = "Cinta-Hilos.jpg" 

# Llamar a la función
conteo_final = procesar_cinta(ruta_del_archivo)

# Mostrar el resultado final
if conteo_final is not None:
    print("\n" + "="*40)
    print(f"✅ RESULTADO FINAL: Se han detectado **{conteo_final}** filamentos.")
    print("="*40)

# Nota: El código requiere que tengas instalados:
# - opencv-python: pip install opencv-python
# - numpy: pip install numpy
# - matplotlib: pip install matplotlib