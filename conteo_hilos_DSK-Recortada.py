import cv2
import numpy as np
import matplotlib.pyplot as plt

def contar_filamentos_laplaciano(ruta_imagen):
    """
    Carga la imagen, aplica el filtro Laplaciano para realzar los bordes/canales, 
    y utiliza la proyección vertical para contar los filamentos.
    """
    try:
        # 1. Cargar la imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"Error: No se pudo cargar la imagen en la ruta: {ruta_imagen}")
            return None
            
        # --- NUEVA LÍNEA: Redimensionar a 450x450 píxeles ---
        # Usamos INTER_AREA para reducir el tamaño, ya que es el método de interpolación recomendado.
        imagen = cv2.resize(imagen, (450, 450), interpolation=cv2.INTER_AREA)
        print("✅ Imagen redimensionada a 450x450 píxeles.")
        # --- FIN NUEVA LÍNEA ---

        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Opcional: Suavizado para reducir ruido antes de la detección de bordes
        suavizada = cv2.GaussianBlur(gris, (3, 3), 0)

        # 2. Detección y Acentuación de Bordes con Laplaciano
        laplaciano = cv2.Laplacian(suavizada, cv2.CV_16S, ksize=3) 
        laplaciano_abs = cv2.convertScaleAbs(laplaciano)

        # 3. Proyección Vertical sobre la imagen Laplaciana
        perfil_vertical = np.sum(laplaciano_abs.astype(np.float32), axis=0)
        
        # Normalizar y convertir a 8 bits
        perfil_normalizado = perfil_vertical / perfil_vertical.max() * 255
        perfil_normalizado = perfil_normalizado.astype(np.uint8)

        # 4. Detección y Conteo de picos
        umbral_pico = 120 
        _, picos = cv2.threshold(perfil_normalizado, umbral_pico, 1, cv2.THRESH_BINARY)
        
        # Usar Clausura para unir partes cercanas (misma lógica que antes)
        kernel_conteo = np.ones((1, 5), np.uint8)
        picos_cerrados = cv2.morphologyEx(picos[np.newaxis, :], cv2.MORPH_CLOSE, kernel_conteo)
        
        # Contar componentes conectados (picos = bordes/canales)
        etiquetas, num_componentes = cv2.connectedComponents(picos_cerrados)
        num_bordes = num_componentes - 1

        # 5. Determinar el número de filamentos
        num_filamentos = num_bordes 
        
        
        # --- Visualización ---
        cv2.imshow("1. Laplaciano (Bordes Acentuados) - 450x450", laplaciano_abs) 
        
        plt.figure(figsize=(10, 4))
        plt.plot(perfil_normalizado, label="Perfil de Intensidad Laplaciana")
        plt.axhline(y=umbral_pico, color='r', linestyle='-', label=f"Umbral ({umbral_pico})")
        plt.plot(picos_cerrados[0] * perfil_normalizado.max(), label="Picos Detectados (Canales)", linestyle='dashed')
        plt.title(f"Perfil Vertical Laplaciano (Picos detectados: {num_bordes})")
        plt.xlabel("Columna de Píxeles")
        plt.ylabel("Suma de Intensidad Normalizada")
        plt.legend()
        plt.grid(axis='y')
        
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return num_filamentos

    except Exception as e:
        print(f"Ocurrió un error durante el procesamiento: {e}")
        return None

# --- Ejecución ---
# Usamos la imagen recortada
ruta_del_archivo = "Cinta-Hilos-Recortada.jpg" 

# Llamar a la función
conteo_final = contar_filamentos_laplaciano(ruta_del_archivo)

if conteo_final is not None:
    print("\n" + "="*40)
    print(f"✅ RESULTADO FINAL con Laplaciano: Se han detectado **{conteo_final}** elementos (picos/canales).")
    print("El número final de filamentos debe ser muy cercano a este valor.")
    print("="*40)