import cv2
import mediapipe as mp
from collections import deque
import numpy as np
import sys
import time

def check_available_cameras():
    """Check all available camera indices"""
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def initialize_camera(camera_index=0, attempts=3):
    """Initialize camera with multiple attempts and detailed error checking"""
    for attempt in range(attempts):
        print(f"Attempting to initialize camera {camera_index} (attempt {attempt + 1}/{attempts})")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Failed to open camera {camera_index}")
            if attempt < attempts - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
                continue
            
            # Check for other available cameras
            available_cameras = check_available_cameras()
            if available_cameras:
                print(f"Available cameras found at indices: {available_cameras}")
                print(f"Try running the program with a different camera index.")
            else:
                print("No available cameras found.")
            
            raise RuntimeError(f"""
Camera initialization failed after {attempts} attempts.
Please check:
1. Camera is properly connected
2. Camera permissions are granted
3. No other application is using the camera
4. Correct camera index is being used
""")
        
        # Test reading a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Camera opened but failed to read frame")
            cap.release()
            if attempt < attempts - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
                continue
            raise RuntimeError("Camera opened but cannot read frames")
        
        # If we got here, camera is working
        print(f"Successfully initialized camera {camera_index}")
        print(f"Frame size: {frame.shape}")
        return cap
    
    raise RuntimeError("Failed to initialize camera after all attempts")

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
    """Classify gesture with validation"""
    if not puntos_normalizados or not historial_puntos:
        return None
        
    gesto_detectado = None
    if len(historial_puntos) == historial_puntos.maxlen and len(puntos_normalizados) == 5:
        try:
            velocidades = []
            for i in range(len(puntos_normalizados)):
                velocidad = np.linalg.norm(np.array(historial_puntos[-1][i]) - np.array(historial_puntos[0][i]))
                velocidades.append(velocidad)
            
            for gesto, valores in gestos.items():
                umbral_velocidad = 5  # Ajustar según sea necesario
                if all(velocidad < umbral_velocidad if valor == 0 else velocidad > umbral_velocidad 
                      for velocidad, valor in zip(velocidades, valores)):
                    gesto_detectado = gesto
                    break
        except Exception as e:
            print(f"Error al clasificar gesto: {e}")
            return None
            
    return gesto_detectado

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
        # Check available cameras first
        available_cameras = check_available_cameras()
        if not available_cameras:
            raise RuntimeError("No cameras available on the system")
        print(f"Available cameras: {available_cameras}")
        
        # Try to initialize the first available camera
        camera_index = available_cameras[0]
        cap = initialize_camera(camera_index)
        
        # Configurar la detección de manos
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            
            print("Iniciando reconocimiento de gestos. Presione 'q' para salir.")
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                # Leer un frame del video
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Error al leer frame de la cámara")
                    print(f"Camera status - isOpened: {cap.isOpened()}")
                    # Try to reinitialize camera
                    cap.release()
                    try:
                        cap = initialize_camera(camera_index)
                        continue
                    except RuntimeError:
                        break

                frame_count += 1
                if frame_count % 30 == 0:  # Calculate FPS every 30 frames
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