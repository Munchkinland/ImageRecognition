import cv2
import torch

# Cargar el modelo YOLOv5 desde el repositorio de Ultralytics con torch.hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Capturar el flujo de vídeo de la cámara web
cap = cv2.VideoCapture(0)
frame_skip = 3  # Procesar cada 3 cuadros para aumentar la velocidad
frame_count = 0
cached_detections = []  # Para almacenar las detecciones entre actualizaciones

while True:
    # Leer un cuadro del flujo de vídeo
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar solo cada `frame_skip` cuadros
    if frame_count % frame_skip == 0:
        # Redimensionar el cuadro para mejorar velocidad
        resized_frame = cv2.resize(frame, (640, 480))

        # Realizar detección con el modelo YOLOv5
        results = model(resized_frame)

        # Extraer detecciones y escalarlas a las dimensiones originales
        detections = results.xyxy[0].cpu().numpy()
        scale_x = frame.shape[1] / resized_frame.shape[1]
        scale_y = frame.shape[0] / resized_frame.shape[0]

        # Actualizar las detecciones almacenadas
        cached_detections = [
            {
                "label": model.names[int(det[5])],
                "score": det[4],
                "box": {
                    "xmin": int(det[0] * scale_x),
                    "ymin": int(det[1] * scale_y),
                    "xmax": int(det[2] * scale_x),
                    "ymax": int(det[3] * scale_y)
                }
            }
            for det in detections if det[4] > 0.8  # Filtrar detecciones con baja confianza
        ]
    
    # Dibujar las detecciones en el cuadro actual
    for detection in cached_detections:
        x1, y1 = detection["box"]["xmin"], detection["box"]["ymin"]
        x2, y2 = detection["box"]["xmax"], detection["box"]["ymax"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
        cv2.putText(frame, f"{detection['label']} {detection['score']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    
    # Mostrar el cuadro de video con detecciones
    cv2.imshow("YOLOv5 Object Detection", frame)
    frame_count += 1

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
