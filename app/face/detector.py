"""
Detector Profissional de Rostos - VERSÃO FINAL
Usa DNN (Deep Neural Network) com fallback para MTCNN/Haar
MUITO mais preciso e confiável
"""
import cv2
import numpy as np
from pathlib import Path

class FaceDetector:
    def __init__(self):
        """Inicializa detector com múltiplos métodos"""
        self.use_dnn = self._load_dnn()
        
        if not self.use_dnn:
            try:
                from mtcnn import MTCNN
                self.mtcnn = MTCNN()
                self.use_mtcnn = True
                print("✓ Detector MTCNN inicializado")
            except:
                self.use_mtcnn = False
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                print("⚠ Usando Haar Cascade (fallback)")
    
    def _load_dnn(self):
        """Carrega modelo DNN do OpenCV"""
        try:
            model_dir = Path(__file__).parent / "models"
            model_dir.mkdir(exist_ok=True)
            
            prototxt = model_dir / "deploy.prototxt"
            caffemodel = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
            
            if not caffemodel.exists():
                print("📥 Baixando modelo DNN (primeira vez)...")
                import urllib.request
                
                proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
                model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
                
                urllib.request.urlretrieve(proto_url, prototxt)
                urllib.request.urlretrieve(model_url, caffemodel)
                print("✓ Modelo DNN baixado")
            
            self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
            print("✓ Detector DNN inicializado (ALTA PRECISÃO)")
            return True
        
        except Exception as e:
            print(f"⚠ DNN não disponível: {e}")
            return False
    
    def detect_faces(self, image):
        """Detecta rostos com alta precisão"""
        if self.use_dnn:
            return self._detect_dnn(image)
        elif self.use_mtcnn:
            return self._detect_mtcnn(image)
        else:
            return self._detect_haar(image)
    
    def _detect_dnn(self, image):
        """Detecção com DNN"""
        h, w = image.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                
                x, y = max(0, x), max(0, y)
                x2, y2 = min(w, x2), min(h, y2)
                
                width = x2 - x
                height = y2 - y
                
                if width > 20 and height > 20:
                    faces.append({
                        'bbox': (x, y, width, height),
                        'confidence': float(confidence),
                        'landmarks': None
                    })
        
        return faces
    
    def _detect_mtcnn(self, image):
        """Detecção com MTCNN"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.mtcnn.detect_faces(rgb)
        
        faces = []
        for detection in detections:
            x, y, w, h = detection['box']
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': detection['confidence'],
                'landmarks': detection['keypoints']
            })
        
        return faces
    
    def _detect_haar(self, image):
        """Detecção com Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces_rects = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        
        faces = []
        for (x, y, w, h) in faces_rects:
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': 0.8,
                'landmarks': None
            })
        
        return faces
    
    def extract_face(self, image, bbox, target_size=(160, 160), margin=0.3):
        """Extrai e redimensiona rosto com margem maior"""
        x, y, w, h = bbox
        
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return np.zeros((*target_size, 3), dtype=np.uint8)
        
        face = cv2.resize(face, target_size, interpolation=cv2.INTER_CUBIC)
        return face
    
    def draw_detections(self, image, faces):
        """Desenha detecções com cores baseadas na confiança"""
        img_copy = image.copy()
        
        for face in faces:
            x, y, w, h = face['bbox']
            conf = face['confidence']
            
            if conf > 0.9:
                color = (0, 255, 0)
            elif conf > 0.7:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)
            
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 2)
            
            text = f"{conf:.1%}"
            cv2.putText(img_copy, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if face['landmarks']:
                for point in face['landmarks'].values():
                    cv2.circle(img_copy, point, 2, (0, 255, 0), -1)
        
        return img_copy
    
    def analyze_quality(self, image, face_bbox):
        """
        Análise RIGOROSA de qualidade da imagem
        Retorna métricas detalhadas e score 0-100
        """
        x, y, w, h = face_bbox
        motivos = []
        metrics = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Luminosidade
        brightness = np.mean(gray)
        metrics['brightness'] = brightness
        
        if brightness < 60:
            motivos.append("iluminação insuficiente")
        elif brightness > 200:
            motivos.append("imagem superexposta")
        
        # 2. Contraste
        contrast = gray.std()
        metrics['contrast'] = contrast
        
        if contrast < 40:
            motivos.append("baixo contraste")
        
        # 3. Nitidez
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = laplacian
        
        if laplacian < 200:
            motivos.append("imagem desfocada")
        
        # 4. Tamanho do rosto
        face_area = w * h
        image_area = image.shape[0] * image.shape[1]
        face_ratio = face_area / image_area
        metrics['face_ratio'] = face_ratio
        
        if face_ratio < 0.08:
            motivos.append("rosto muito pequeno")
        elif face_ratio > 0.6:
            motivos.append("rosto muito próximo")
        
        # 5. Resolução
        if w < 100 or h < 100:
            motivos.append("resolução facial baixa")
        
        # 6. Proporção
        aspect_ratio = w / h
        metrics['aspect_ratio'] = aspect_ratio
        
        if aspect_ratio < 0.75 or aspect_ratio > 1.3:
            motivos.append("proporção facial anormal")
        
        # 7. Bordas
        margin = 10
        if x < margin or y < margin or \
           x + w > image.shape[1] - margin or \
           y + h > image.shape[0] - margin:
            motivos.append("rosto cortado nas bordas")
        
        # Score de qualidade (0-100)
        quality_score = 0
        
        if 60 <= brightness <= 200:
            quality_score += 20
        if contrast >= 40:
            quality_score += 20
        if laplacian >= 200:
            quality_score += 25
        if 0.08 <= face_ratio <= 0.6:
            quality_score += 20
        if 0.75 <= aspect_ratio <= 1.3:
            quality_score += 15
        
        metrics['quality_score'] = quality_score
        
        return {
            **metrics,
            'motivos': motivos,
            'is_good_quality': quality_score >= 70
        }