"""
Detector de rostos usando Haar Cascade (OpenCV)
Simples, rápido e funcional
"""
import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        """Inicializa detector"""
        # Carregar Haar Cascade (já vem com OpenCV)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise Exception("Erro ao carregar Haar Cascade")
        
        print("✓ Detector de rostos inicializado")
    
    def detect_faces(self, image):
        """
        Detecta rostos na imagem
        Retorna lista de rostos detectados
        """
        # Converter para cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Detectar rostos
        faces_rects = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        faces = []
        for (x, y, w, h) in faces_rects:
            # Garantir limites
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            # Confiança aproximada
            face_area = w * h
            image_area = image.shape[0] * image.shape[1]
            confidence = min(0.95, (face_area / image_area) * 8)
            
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': confidence
            })
        
        return faces
    
    def extract_face(self, image, bbox, target_size=(160, 160), margin=0.2):
        """
        Extrai rosto da imagem e redimensiona
        """
        x, y, w, h = bbox
        
        # Adicionar margem
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        # Recortar
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return np.zeros((*target_size, 3), dtype=np.uint8)
        
        # Redimensionar
        face = cv2.resize(face, target_size)
        return face
    
    def draw_detections(self, image, faces):
        """
        Desenha retângulos nos rostos
        """
        img_copy = image.copy()
        
        for face in faces:
            x, y, w, h = face['bbox']
            conf = face['confidence']
            
            # Retângulo verde
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Texto da confiança
            text = f"{conf:.0%}"
            cv2.putText(img_copy, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return img_copy
    
    def analyze_quality(self, image, face_bbox):
        """
        Analisa qualidade da imagem
        """
        motivos = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Luminosidade
        brightness = np.mean(gray)
        if brightness < 50:
            motivos.append("muito escuro")
        elif brightness > 200:
            motivos.append("muito claro")
        
        # Nitidez
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < 100:
            motivos.append("imagem desfocada")
        
        # Tamanho do rosto
        x, y, w, h = face_bbox
        face_ratio = (w * h) / (image.shape[0] * image.shape[1])
        if face_ratio < 0.05:
            motivos.append("rosto muito pequeno")
        
        return {
            "brightness": brightness,
            "sharpness": sharpness,
            "face_ratio": face_ratio,
            "motivos": motivos
        }