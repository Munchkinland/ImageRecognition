import cv2
import mediapipe as mp
from collections import deque
import numpy as np

# Inicializar Mediapipe
mp_hands = mp.solutions.hands

# Función para detectar manos en una imagen
def detectar_manos(imagen, hands):
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultados = hands.process(imagen_rgb)
    return resultados

# Función para obtener puntos de referencia de la mano
def obtener_puntos_referencia(imagen, resultados):
    puntos_normalizados = []
    if resultados.multi_hand_landmarks:
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

# Función para clasificar el gesto
def clasificar_gesto(puntos_normalizados, gestos, historial_puntos):
    gesto_detectado = None
    if len(historial_puntos) == historial_puntos.maxlen and len(puntos_normalizados) == 5:
        velocidades = []
        for i in range(len(puntos_normalizados)):
            velocidad = np.linalg.norm(np.array(historial_puntos[-1][i]) - np.array(historial_puntos[0][i]))
            velocidades.append(velocidad)
        for gesto, valores in gestos.items():
            umbral_velocidad = 5  # Ajustar según sea necesario
            if all(velocidad < umbral_velocidad if valor == 0 else velocidad > umbral_velocidad for velocidad, valor in zip(velocidades, valores)):
                gesto_detectado = gesto
                break
    return gesto_detectado

# Función para mostrar el resultado
def mostrar_resultado(imagen, gesto_detectado, puntos_normalizados, resultados):
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(imagen, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    if gesto_detectado:
        cv2.putText(imagen, gesto_detectado, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return imagen

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

# Configurar la detección de manos
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    # Iniciar la captura de video
    cap = cv2.VideoCapture(0)

    while True:
        # Leer un frame del video
        ret, frame = cap.read()

        # Detectar manos
        resultados = detectar_manos(frame, hands)

        # Obtener puntos de referencia
        puntos_normalizados = obtener_puntos_referencia(frame, resultados)

        # Agregar puntos al historial
        historial_puntos.append(puntos_normalizados)

        # Clasificar gesto
        gesto = clasificar_gesto(puntos_normalizados, gestos, historial_puntos)

        # Mostrar resultado
        imagen_procesada = mostrar_resultado(frame, gesto, puntos_normalizados, resultados)

        # Mostrar la imagen procesada
        cv2.imshow('Reconocimiento de Gestos', imagen_procesada)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y destruir las ventanas
    cap.release()
    cv2.destroyAllWindows()