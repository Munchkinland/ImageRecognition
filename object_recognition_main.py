import cv2
import torch
import numpy as np
from typing import List, Dict
import time

class ObjectDetector:
    def __init__(self, model_name: str = 'yolov5s', conf_threshold: float = 0.45):
        """
        Inicializa el detector de objetos
        """
        self.conf_threshold = conf_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Cargar el modelo
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Optimizaciones para inference
        self.model.conf = conf_threshold  # Umbral de confianza
        self.model.iou = 0.45  # Umbral IoU para NMS
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # Una etiqueta por box
        self.model.max_det = 50  # Máximo número de detecciones

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Procesa un frame y retorna las detecciones
        """
        # Convertir a RGB y realizar la inferencia
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Realizar inferencia
        results = self.model(frame_rgb, size=640)  # Tamaño de inferencia fijo para mejor rendimiento
        
        # Procesar resultados
        detections = []
        if len(results.pred[0]) > 0:
            for *xyxy, conf, cls in results.pred[0].cpu().numpy():
                if conf >= self.conf_threshold:
                    detections.append({
                        "label": self.model.names[int(cls)],
                        "score": float(conf),
                        "box": {
                            "xmin": int(xyxy[0]),
                            "ymin": int(xyxy[1]),
                            "xmax": int(xyxy[2]),
                            "ymax": int(xyxy[3])
                        }
                    })
        
        return detections

class VideoProcessor:
    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720):
        """
        Inicializa el procesador de video
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def get_frame(self) -> tuple:
        """
        Obtiene un frame de la cámara
        """
        return self.cap.read()

    def release(self):
        """
        Libera los recursos
        """
        self.cap.release()

def draw_detections(frame: np.ndarray, detections: List[Dict], fps: float = 0.0) -> np.ndarray:
    """
    Dibuja las detecciones en el frame
    """
    # Crear una copia del frame para no modificar el original
    output_frame = frame.copy()
    
    # Dibujar cada detección
    for det in detections:
        box = det["box"]
        label = det["label"]
        score = det["score"]
        
        # Color único para cada clase
        color = (
            hash(label) % 256,
            hash(label * 2) % 256,
            hash(label * 3) % 256
        )
        
        # Dibujar bbox
        cv2.rectangle(output_frame,
                     (box["xmin"], box["ymin"]),
                     (box["xmax"], box["ymax"]),
                     color, 2)
        
        # Texto con label y score
        text = f'{label} {score:.2f}'
        font_scale = 0.6
        thickness = 2
        
        # Obtener tamaño del texto
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Dibujar fondo para el texto
        cv2.rectangle(output_frame,
                     (box["xmin"], box["ymin"] - text_height - 5),
                     (box["xmin"] + text_width, box["ymin"]),
                     color, -1)
        
        # Dibujar texto
        cv2.putText(output_frame, text,
                    (box["xmin"], box["ymin"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    thickness)
    
    # Mostrar FPS
    cv2.putText(output_frame, f'FPS: {fps:.1f}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return output_frame

def main():
    # Inicializar detector y procesador de video
    detector = ObjectDetector(model_name='yolov5s', conf_threshold=0.45)
    video = VideoProcessor()
    
    frame_count = 0
    fps = 0
    fps_start_time = time.time()
    
    print("Presiona 'q' para salir")
    
    while True:
        ret, frame = video.get_frame()
        if not ret:
            break
            
        # Procesar frame y obtener detecciones
        detections = detector.process_frame(frame)
        
        # Calcular FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
        
        # Dibujar resultados
        output_frame = draw_detections(frame, detections, fps)
        
        # Mostrar resultado
        cv2.imshow('YOLOv5 Multiple Object Detection', output_frame)
        
        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()