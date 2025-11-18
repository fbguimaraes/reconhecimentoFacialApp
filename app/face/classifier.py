"""
Classificador PROFISSIONAL com Ensemble - VERSÃO FINAL
Combina SVM + KNN + Distance para máxima precisão
Thresholds personalizados por pessoa
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import pickle
from pathlib import Path
from collections import Counter
from .embeddings import FaceEmbeddings
from ..utils.config import DATA_DIR, MIN_CONFIDENCE

class FaceClassifier:
    def __init__(self):
        """Inicializa classificador ensemble profissional"""
        self.embedder = FaceEmbeddings()
        
        # Ensemble de classificadores
        self.svm = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
        self.knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')
        self.scaler = StandardScaler()
        
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Thresholds personalizados
        self.person_thresholds = {}
        
        self.model_path = DATA_DIR / "model_ensemble.pkl"
        self.encoder_path = DATA_DIR / "encoder_ensemble.pkl"
        self.scaler_path = DATA_DIR / "scaler_ensemble.pkl"
        self.thresholds_path = DATA_DIR / "thresholds_ensemble.pkl"
        
        self._load_model()
    
    def _load_model(self):
        """Carrega modelos salvos"""
        try:
            if all(p.exists() for p in [self.model_path, self.encoder_path, self.scaler_path]):
                with open(self.model_path, 'rb') as f:
                    models = pickle.load(f)
                    self.svm = models['svm']
                    self.knn = models['knn']
                
                with open(self.encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                if self.thresholds_path.exists():
                    with open(self.thresholds_path, 'rb') as f:
                        self.person_thresholds = pickle.load(f)
                
                self.is_trained = True
                print(f"✓ Modelo ENSEMBLE carregado: {len(self.label_encoder.classes_)} pessoas")
        
        except Exception as e:
            print(f"⚠ Erro ao carregar modelo: {e}")
            self.is_trained = False
    
    def _save_model(self):
        """Salva todos os modelos"""
        with open(self.model_path, 'wb') as f:
            pickle.dump({'svm': self.svm, 'knn': self.knn}, f)
        
        with open(self.encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(self.thresholds_path, 'wb') as f:
            pickle.dump(self.person_thresholds, f)
        
        print("✓ Modelo ENSEMBLE salvo")
    
    def register_person(self, name, face_images):
        """
        Registra pessoa com augmentação automática
        Gera múltiplos embeddings por imagem para robustez
        """
        print(f"\n📸 Registrando '{name}' - Processamento profissional...")
        
        all_embeddings = []
        
        for i, img in enumerate(face_images):
            print(f"  Imagem {i+1}/{len(face_images)}: ", end="")
            
            # Gerar embeddings (original + augmentations)
            embeddings = self.embedder.get_multiple_embeddings(img, augment=True)
            
            valid_count = 0
            for emb in embeddings:
                is_valid, msg = self.embedder.validate_embedding(emb)
                if is_valid:
                    all_embeddings.append(emb)
                    valid_count += 1
            
            print(f"✓ {valid_count} embeddings válidos")
        
        if len(all_embeddings) == 0:
            print("❌ Nenhum embedding válido!")
            return False
        
        # Salvar embeddings
        person_dir = DATA_DIR / name
        person_dir.mkdir(exist_ok=True)
        
        for i, emb in enumerate(all_embeddings):
            emb_path = person_dir / f"emb_{i}.npy"
            np.save(emb_path, emb)
        
        print(f"✓ Total: {len(all_embeddings)} embeddings salvos")
        print(f"  (Média: {len(all_embeddings)/len(face_images):.1f} por imagem)")
        
        # Retreinar
        return self.train()
    
    def train(self):
        """
        Treina ensemble com validação cruzada
        """
        print("\n🎓 Treinando ENSEMBLE de classificadores...")
        
        X = []
        y = []
        
        # Carregar embeddings
        for person_dir in DATA_DIR.iterdir():
            if not person_dir.is_dir():
                continue
            
            name = person_dir.name
            count = 0
            
            for emb_file in person_dir.glob("emb_*.npy"):
                emb = self.embedder.load_embedding(emb_file)
                if emb is not None:
                    X.append(emb)
                    y.append(name)
                    count += 1
            
            if count > 0:
                print(f"  {name}: {count} embeddings")
        
        if len(X) == 0:
            print("❌ Nenhum embedding para treinar")
            return False
        
        # Se só tem 1 pessoa, criar classe negativa
        if len(set(y)) < 2:
            print(f"⚠ Apenas 1 pessoa cadastrada")
            print("  Criando amostras negativas artificiais...")
            
            for i in range(len(X)):
                fake_emb = np.random.randn(128) * 0.3
                fake_emb = fake_emb / (np.linalg.norm(fake_emb) + 1e-10)
                X.append(fake_emb)
                y.append("_unknown_")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n📊 Dataset: {len(X)} amostras, {len(set(y))} classes")
        
        # Codificar labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Escalar
        X_scaled = self.scaler.fit_transform(X)
        
        # Treinar SVM
        print("  [1/3] Treinando SVM...", end=" ")
        self.svm.fit(X_scaled, y_encoded)
        
        if len(X) >= 3:
            svm_score = cross_val_score(self.svm, X_scaled, y_encoded, cv=min(3, len(X))).mean()
            print(f"✓ CV Score: {svm_score:.1%}")
        else:
            print("✓")
        
        # Treinar KNN
        print("  [2/3] Treinando KNN...", end=" ")
        self.knn.fit(X_scaled, y_encoded)
        
        if len(X) >= 3:
            knn_score = cross_val_score(self.knn, X_scaled, y_encoded, cv=min(3, len(X))).mean()
            print(f"✓ CV Score: {knn_score:.1%}")
        else:
            print("✓")
        
        # Calcular thresholds personalizados
        print("  [3/3] Calibrando thresholds...", end=" ")
        self._calculate_person_thresholds(X, y)
        print("✓")
        
        self.is_trained = True
        self._save_model()
        
        print(f"\n✅ Treinamento COMPLETO!")
        print(f"   Ensemble pronto: {len(set(y))} pessoas")
        
        return True
    
    def _calculate_person_thresholds(self, X, y):
        """
        Calcula threshold ótimo para cada pessoa
        Baseado na variabilidade intra-classe
        """
        unique_people = [p for p in set(y) if p != "_unknown_"]
        
        for person in unique_people:
            person_embeddings = X[y == person]
            
            if len(person_embeddings) < 2:
                self.person_thresholds[person] = 0.6
                continue
            
            # Distâncias intra-classe
            distances = []
            for i in range(len(person_embeddings)):
                for j in range(i+1, len(person_embeddings)):
                    dist = self.embedder.compute_distance(
                        person_embeddings[i],
                        person_embeddings[j]
                    )
                    distances.append(dist)
            
            # Threshold = média + 1.5*desvio
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            threshold = min(0.8, mean_dist + 1.5 * std_dist)
            
            self.person_thresholds[person] = threshold
    
    def recognize(self, face_image, threshold=None):
        """
        Reconhecimento PROFISSIONAL com ensemble voting
        Combina SVM + KNN + Distance para decisão final
        """
        if threshold is None:
            threshold = MIN_CONFIDENCE
        
        if not self.is_trained:
            return {
                "nome": "desconhecido",
                "confianca": 0.0,
                "motivos": ["sistema não treinado"],
                "embedding_dist": None,
                "quality_score": 0,
                "ensemble_votes": {}
            }
        
        # Gerar embedding
        embedding = self.embedder.get_embedding(face_image, num_jitters=10, model='large')
        
        # Validar
        is_valid, msg = self.embedder.validate_embedding(embedding)
        if not is_valid:
            return {
                "nome": "desconhecido",
                "confianca": 0.0,
                "motivos": [f"embedding inválido: {msg}"],
                "embedding_dist": None,
                "quality_score": 0,
                "ensemble_votes": {}
            }
        
        embedding_scaled = self.scaler.transform(embedding.reshape(1, -1))
        
        # 1. SVM
        svm_proba = self.svm.predict_proba(embedding_scaled)[0]
        svm_class = np.argmax(svm_proba)
        svm_conf = svm_proba[svm_class]
        svm_name = self.label_encoder.inverse_transform([svm_class])[0]
        
        # 2. KNN
        knn_proba = self.knn.predict_proba(embedding_scaled)[0]
        knn_class = np.argmax(knn_proba)
        knn_conf = knn_proba[knn_class]
        knn_name = self.label_encoder.inverse_transform([knn_class])[0]
        
        # 3. Distance-based
        all_embeddings = []
        all_labels = []
        
        for person_dir in DATA_DIR.iterdir():
            if not person_dir.is_dir():
                continue
            for emb_file in person_dir.glob("emb_*.npy"):
                emb = self.embedder.load_embedding(emb_file)
                if emb is not None:
                    all_embeddings.append(emb)
                    all_labels.append(person_dir.name)
        
        dist_name, min_distance, dist_conf = self.embedder.get_best_match(
            all_embeddings, all_labels, embedding, tolerance=0.6
        )
        
        # Ensemble Voting
        votes = Counter([svm_name, knn_name, dist_name])
        ensemble_name = votes.most_common(1)[0][0]
        
        # Confiança combinada (média ponderada)
        if ensemble_name == svm_name:
            combined_conf = (svm_conf * 0.4 + knn_conf * 0.3 + dist_conf * 0.3)
        elif ensemble_name == knn_name:
            combined_conf = (svm_conf * 0.3 + knn_conf * 0.4 + dist_conf * 0.3)
        else:
            combined_conf = dist_conf
        
        # Threshold personalizado
        person_threshold = self.person_thresholds.get(ensemble_name, 0.6)
        
        # Avaliar
        motivos = []
        
        if combined_conf < threshold:
            motivos.append(f"confiança abaixo do mínimo ({combined_conf:.1%} < {threshold:.1%})")
        
        if min_distance and min_distance > person_threshold:
            motivos.append(f"distância facial elevada ({min_distance:.3f})")
        
        # Verificar concordância
        if len(set([svm_name, knn_name, dist_name])) == 3:
            motivos.append("classificadores discordam - incerteza alta")
            combined_conf *= 0.7
        
        # Decisão final
        if ensemble_name == "_unknown_" or combined_conf < threshold or len(motivos) > 0:
            final_name = "desconhecido"
        else:
            final_name = ensemble_name
        
        return {
            "nome": final_name,
            "confianca": float(combined_conf),
            "motivos": motivos,
            "embedding_dist": float(min_distance) if min_distance else None,
            "quality_score": 0,
            "ensemble_votes": {
                "svm": {"nome": svm_name, "conf": float(svm_conf)},
                "knn": {"nome": knn_name, "conf": float(knn_conf)},
                "distance": {"nome": dist_name if dist_name else "N/A", "conf": float(dist_conf)}
            }
        }
    
    def list_people(self):
        """Lista pessoas com estatísticas"""
        people = []
        
        for person_dir in DATA_DIR.iterdir():
            if person_dir.is_dir() and person_dir.name != "_unknown_":
                embeddings = list(person_dir.glob("emb_*.npy"))
                threshold = self.person_thresholds.get(person_dir.name, 0.6)
                
                people.append({
                    "nome": person_dir.name,
                    "num_embeddings": len(embeddings),
                    "threshold_personalizado": round(threshold, 3)
                })
        
        return people
    
    def delete_person(self, name):
        """Remove pessoa e retreina"""
        person_dir = DATA_DIR / name
        if person_dir.exists():
            import shutil
            shutil.rmtree(person_dir)
            
            if name in self.person_thresholds:
                del self.person_thresholds[name]
            
            self.train()
            return True
        return False
    
    def get_model_stats(self):
        """Estatísticas detalhadas do modelo"""
        if not self.is_trained:
            return {"treinado": False}
        
        return {
            "treinado": True,
            "num_pessoas": len(self.label_encoder.classes_),
            "pessoas": list(self.label_encoder.classes_),
            "thresholds_personalizados": {k: round(v, 3) for k, v in self.person_thresholds.items()},
            "classificadores": ["SVM (RBF)", "KNN (k=3)", "Distance-based"],
            "metodo": "Ensemble Voting"
        }