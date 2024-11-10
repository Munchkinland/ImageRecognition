import cv2
import mediapipe as mp
from collections import deque
import numpy as np
import sys
import time
import platform

def initialize_camera_mac():
    """Initialize camera specifically for MacOS"""
    # Try different camera APIs
    apis = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    
    for api in apis:
        print(f"Intentando inicializar cámara con API: {api}")
        cap = cv2.VideoCapture(0 + api)
        
        if not cap.isOpened():
            print(f"No se pudo abrir la cámara con API {api}")
            continue
            
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Test reading a frame
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Cámara inicializada exitosamente con API {api}")
            print(f"Resolución: {frame.shape}")
            return cap
            
        cap.release()
    
    raise RuntimeError("""
No se pudo inicializar la cámara. Por favor verifica:
1. Que la cámara está conectada
2. Que has dado permisos de cámara a la Terminal/IDE
   - Ve a Preferencias del Sistema > Seguridad y Privacidad > Privacidad > Cámara
   - Activa el permiso para Terminal o tu IDE
3. Que ninguna otra aplicación está usando la cámara
""")

def detectar_manos(imagen, hands):
    """Detect hands in image with error handling"""
    if imagen is None or imagen.size == 0:
        return None
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultados = hands.process(imagen_rgb)
    return resultados

def obtener_puntos_referencia(imagen, resultados):
    """Get hand landmarks with proper error handling"""
    puntos_normalizados = []
    if resultados and resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            puntos = []
            for id, landmark in enumerate(hand_landmarks.landmark):
                if id in [4, 8, 12, 16, 20]:  # IDs de las puntas de los dedos
                    puntos.append(landmark)
            for punto in puntos:
                x = int(punto.x * imagen.shape[1])
                y = int(punto.y * imagen.shape[0])
                puntos_normalizados.append((x, y))
    return puntos_normalizados

def clasificar_gesto(puntos_normalizados, gestos, historial_puntos):
    """
    Classify hand gesture based on finger positions and movement
    Args:
        puntos_normalizados: List of normalized finger tip coordinates
        gestos: Dictionary of gesture definitions
        historial_puntos: Deque of historical points
    Returns:
        str: Detected gesture name or None
    """
    if not puntos_normalizados or not historial_puntos:
        return None
        
    if len(historial_puntos) == historial_puntos.maxlen and len(puntos_normalizados) == 5:
        try:
            # Calcular posiciones relativas de los dedos respecto al pulgar
            pulgar = np.array(puntos_normalizados[0])
            dedos_levantados = []
            
            # Por cada dedo (excluyendo el pulgar)
            for i in range(1, 5):
                dedo = np.array(puntos_normalizados[i])
                
                # Calcular la distancia vertical entre la punta del dedo y el pulgar
                distancia_vertical = pulgar[1] - dedo[1]
                
                # Un dedo se considera "levantado" si está más arriba que el pulgar
                # y la distancia vertical es significativa
                umbral_distancia = 30
                dedo_levantado = distancia_vertical > umbral_distancia
                dedos_levantados.append(1 if dedo_levantado else 0)
            
            # El pulgar se procesa de manera diferente
            # Se considera levantado si está significativamente a la izquierda o derecha
            dedo_pulgar = np.array(puntos_normalizados[0])
            centro_mano = np.mean([np.array(p) for p in puntos_normalizados[1:]], axis=0)
            distancia_horizontal = abs(dedo_pulgar[0] - centro_mano[0])
            pulgar_levantado = distancia_horizontal > 50
            dedos_estado = [1 if pulgar_levantado else 0] + dedos_levantados
            
            # Comparar con los gestos definidos
            for gesto, valores in gestos.items():
                if dedos_estado == valores:
                    return gesto
                    
        except Exception as e:
            print(f"Error al clasificar gesto: {e}")
            return None
            
    return None

def mostrar_resultado(imagen, gesto_detectado, puntos_normalizados, resultados, mp_hands, mp_drawing=None):
    """Display results with error handling"""
    if imagen is None:
        return None
        
    imagen_output = imagen.copy()
    
    if resultados and resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                imagen_output, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
    
    if gesto_detectado:
        cv2.putText(
            imagen_output,
            gesto_detectado,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    
    return imagen_output

def main():
    # Verificar sistema operativo
    if platform.system() != "Darwin":
        print("Este script está optimizado para macOS")
        sys.exit(1)

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Definir los gestos a reconocer
    gestos = {
        "puño": [0, 0, 0, 0, 0],
        "mano abierta": [1, 1, 1, 1, 1],
        "pulgar arriba": [1, 0, 0, 0, 0],
        "índice arriba": [0, 1, 0, 0, 0],
        "dos dedos": [0, 1, 1, 0, 0],
        "tres dedos": [0, 1, 1, 1, 0],
        "cuatro dedos": [0, 1, 1, 1, 1],
        "ok": [1, 0, 0, 1, 1],
        "pistola": [0, 1, 0, 0, 1],
        "rock": [0, 1, 1, 0, 1],
    }

    # Inicializar la cola para el historial de puntos de referencia
    historial_puntos = deque(maxlen=10)

    try:
        # Inicializar la cámara usando el método específico para macOS
        cap = initialize_camera_mac()
        
        # Configurar la detección de manos con umbrales ajustados
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,  # Aumentado para mayor precisión
            min_tracking_confidence=0.7    # Aumentado para mayor precisión
        ) as hands:
            
            print("Iniciando reconocimiento de gestos. Presione 'q' para salir.")
            
            frame_count = 0
            start_time = time.time()
            last_frame_time = time.time()
            
            while True:
                # Control de FPS para macOS
                current_time = time.time()
                if (current_time - last_frame_time) < 1/30.0:  # Limitar a 30 FPS
                    continue
                last_frame_time = current_time
                
                # Leer un frame del video
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Error al leer frame de la cámara")
                    print("Intentando reiniciar la cámara...")
                    cap.release()
                    try:
                        cap = initialize_camera_mac()
                        continue
                    except RuntimeError as e:
                        print(f"No se pudo reiniciar la cámara: {e}")
                        break

                frame_count += 1
                if frame_count % 30 == 0:  # Calcular FPS cada 30 frames
                    fps = frame_count / (time.time() - start_time)
                    print(f"FPS: {fps:.2f}")

                # Detectar manos
                resultados = detectar_manos(frame, hands)

                # Obtener puntos de referencia
                puntos_normalizados = obtener_puntos_referencia(frame, resultados)

                # Agregar puntos al historial si existen
                if puntos_normalizados:
                    historial_puntos.append(puntos_normalizados)

                # Clasificar gesto
                gesto = clasificar_gesto(puntos_normalizados, gestos, historial_puntos)

                # Mostrar resultado
                imagen_procesada = mostrar_resultado(frame, gesto, puntos_normalizados, resultados, mp_hands, mp_drawing)
                if imagen_procesada is not None:
                    cv2.imshow('Reconocimiento de Gestos', imagen_procesada)

                # Salir si se presiona la tecla 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Limpieza
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Programa finalizado")

if __name__ == "__main__":
    main()