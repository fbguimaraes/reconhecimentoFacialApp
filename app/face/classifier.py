"""
Classificador de rostos usando SVM
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
from .embeddings import FaceEmbeddings
from ..utils.config import DATA_DIR, MIN_CONFIDENCE

class FaceClassifier:
    def __init__(self):
        """Inicializa classificador"""
        self.embedder = FaceEmbeddings()
        self.svm = SVC(kernel='linear', probability=True, C=1.0)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        self.model_path = DATA_DIR / "model.pkl"
        self.encoder_path = DATA_DIR / "encoder.pkl"
        
        self._load_model()
    
    def _load_model(self):
        """Carrega modelo salvo"""
        if self.model_path.exists() and self.encoder_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.svm = pickle.load(f)
                with open(self.encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                self.is_trained = True
                print(f"✓ Modelo carregado: {len(self.label_encoder.classes_)} pessoas")
            except Exception as e:
                print(f"⚠ Erro ao carregar modelo: {e}")
                self.is_trained = False
    
    def _save_model(self):
        """Salva modelo"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.svm, f)
        with open(self.encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print("✓ Modelo salvo")
    
    def register_person(self, name, face_images):
        """
        Registra nova pessoa
        
        Args:
            name: nome da pessoa
            face_images: lista de imagens de rosto (min 3)
        """
        if len(face_images) < 3:
            print(f"⚠ Recomendado 3+ imagens, recebido: {len(face_images)}")
        
        embeddings = []
        for img in face_images:
            emb = self.embedder.get_embedding(img)
            embeddings.append(emb)
        
        # Salvar embeddings
        person_dir = DATA_DIR / name
        person_dir.mkdir(exist_ok=True)
        
        for i, emb in enumerate(embeddings):
            emb_path = person_dir / f"emb_{i}.npy"
            np.save(emb_path, emb)
        
        print(f"✓ {len(embeddings)} embeddings salvos para '{name}'")
        
        # Retreinar
        return self.train()
    
    def train(self):
        """
        Treina classificador SVM
        """
        X = []
        y = []
        
        # Carregar todos os embeddings
        for person_dir in DATA_DIR.iterdir():
            if not person_dir.is_dir():
                continue
            
            name = person_dir.name
            for emb_file in person_dir.glob("emb_*.npy"):
                embedding = np.load(emb_file)
                X.append(embedding)
                y.append(name)
        
        if len(X) < 2:
            print("⚠ Mínimo 2 pessoas para treinar")
            return False
        
        if len(set(y)) < 2:
            print("⚠ Mínimo 2 pessoas diferentes")
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        # Treinar
        y_encoded = self.label_encoder.fit_transform(y)
        self.svm.fit(X, y_encoded)
        self.is_trained = True
        
        self._save_model()
        
        print(f"✓ Treinado: {len(X)} amostras, {len(set(y))} pessoas")
        return True
    
    def recognize(self, face_image, threshold=None):
        """
        Reconhece pessoa
        
        Returns:
            dict com resultado
        """
        if threshold is None:
            threshold = MIN_CONFIDENCE
        
        if not self.is_trained:
            return {
                "nome": "desconhecido",
                "confianca": 0.0,
                "motivos": ["sistema não treinado"],
                "embedding_dist": None
            }
        
        # Gerar embedding
        embedding = self.embedder.get_embedding(face_image)
        embedding = embedding.reshape(1, -1)
        
        # Prever
        proba = self.svm.predict_proba(embedding)[0]
        class_idx = np.argmax(proba)
        confidence = proba[class_idx]
        
        # Calcular distância mínima
        all_embeddings = []
        for person_dir in DATA_DIR.iterdir():
            if not person_dir.is_dir():
                continue
            for emb_file in person_dir.glob("emb_*.npy"):
                emb = np.load(emb_file)
                all_embeddings.append(emb)
        
        if all_embeddings:
            all_embeddings = np.array(all_embeddings)
            distances = [self.embedder.compute_distance(embedding[0], e) 
                        for e in all_embeddings]
            min_distance = min(distances)
        else:
            min_distance = 999
        
        # Avaliar
        motivos = []
        
        if confidence < threshold:
            motivos.append(f"confiança baixa ({confidence:.1%})")
        
        if min_distance > 1.2:
            motivos.append("embedding muito diferente")
        
        # Resultado
        if confidence >= threshold and min_distance <= 1.2:
            nome = self.label_encoder.inverse_transform([class_idx])[0]
        else:
            nome = "desconhecido"
        
        return {
            "nome": nome,
            "confianca": float(confidence),
            "motivos": motivos,
            "embedding_dist": float(min_distance)
        }
    
    def list_people(self):
        """Lista pessoas cadastradas"""
        people = []
        for person_dir in DATA_DIR.iterdir():
            if person_dir.is_dir():
                count = len(list(person_dir.glob("emb_*.npy")))
                people.append({
                    "nome": person_dir.name,
                    "num_embeddings": count
                })
        return people
    
    def delete_person(self, name):
        """Remove pessoa"""
        person_dir = DATA_DIR / name
        if person_dir.exists():
            import shutil
            shutil.rmtree(person_dir)
            self.train()
            return True
        return False