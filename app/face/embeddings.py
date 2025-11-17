"""
Geração de embeddings faciais
Usa HOG + histograma de cores (simples e eficaz)
"""
import cv2
import numpy as np
from pathlib import Path

class FaceEmbeddings:
    def __init__(self):
        """Inicializa gerador de embeddings"""
        self.feature_size = 128
        print("✓ Gerador de embeddings inicializado")
    
    def get_embedding(self, face_img):
        """
        Gera embedding (vetor de características) do rosto
        
        Args:
            face_img: imagem do rosto (160x160 BGR)
            
        Returns:
            embedding: vetor numpy de 128 dimensões
        """
        # Redimensionar para tamanho padrão
        face = cv2.resize(face_img, (64, 64))
        
        # 1. HOG Features (gradientes)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor(
            (64, 64),    # winSize
            (16, 16),    # blockSize
            (8, 8),      # blockStride
            (8, 8),      # cellSize
            9            # nbins
        )
        hog_features = hog.compute(gray).flatten()
        
        # 2. Histograma de cores
        color_hist = []
        for channel in range(3):  # B, G, R
            hist = cv2.calcHist([face], [channel], None, [32], [0, 256])
            color_hist.extend(hist.flatten())
        color_hist = np.array(color_hist)
        
        # 3. Estatísticas básicas
        mean_vals = face.mean(axis=(0, 1))
        std_vals = face.std(axis=(0, 1))
        stats = np.concatenate([mean_vals, std_vals])
        
        # Combinar features
        # HOG: ~1000 dims -> pegar primeiras 64
        # Color: 96 dims
        # Stats: 6 dims
        # Total: aproximadamente 128 dims
        
        hog_reduced = hog_features[:64]
        color_reduced = color_hist[:58]
        
        embedding = np.concatenate([
            hog_reduced,
            color_reduced,
            stats
        ])
        
        # Normalizar para [0, 1]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-7)
        
        # Garantir tamanho exato
        if len(embedding) > self.feature_size:
            embedding = embedding[:self.feature_size]
        elif len(embedding) < self.feature_size:
            padding = np.zeros(self.feature_size - len(embedding))
            embedding = np.concatenate([embedding, padding])
        
        return embedding.astype(np.float32)
    
    def compute_distance(self, emb1, emb2):
        """
        Distância euclidiana entre embeddings
        Quanto menor, mais similar
        """
        return np.linalg.norm(emb1 - emb2)
    
    def compute_similarity(self, emb1, emb2):
        """
        Similaridade cosine [0, 1]
        Quanto maior, mais similar
        """
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        sim = dot / (norm1 * norm2 + 1e-7)
        return (sim + 1) / 2  # Converter para [0, 1]
    
    def save_embedding(self, embedding, filepath):
        """Salva embedding em arquivo"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, embedding)
    
    def load_embedding(self, filepath):
        """Carrega embedding"""
        return np.load(filepath)