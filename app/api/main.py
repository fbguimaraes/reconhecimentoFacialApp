"""
API FastAPI PROFISSIONAL - VERSÃO FINAL
WebSocket + REST + Métricas em tempo real
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import List
import cv2
import numpy as np
import base64
from datetime import datetime
import asyncio
import time

from ..face.detector import FaceDetector
from ..face.classifier import FaceClassifier
from ..utils.logger import access_logger
from ..utils.config import system_status, MIN_CONFIDENCE

app = FastAPI(
    title="Sistema de Reconhecimento Facial PROFISSIONAL",
    version="2.0",
    description="API com Ensemble Learning + Métricas Avançadas"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar componentes profissionais
detector = FaceDetector()
classifier = FaceClassifier()

# Métricas de performance
performance_metrics = {
    "total_reconhecimentos": 0,
    "tempo_medio_ms": 0,
    "precisao_estimada": 0.0,
    "ultimo_tempo_ms": 0
}

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✓ WebSocket: {len(self.active_connections)} conexões ativas")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"✗ WebSocket: {len(self.active_connections)} conexões ativas")
    
    async def broadcast(self, message: dict):
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except:
                dead.append(conn)
        
        for conn in dead:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

manager = ConnectionManager()

@app.get("/")
def root():
    """Informações da API"""
    return {
        "message": "Sistema de Reconhecimento Facial PROFISSIONAL",
        "version": "2.0",
        "status": "online",
        "features": [
            "Ensemble Learning (SVM + KNN + Distance)",
            "Face Recognition (dlib)",
            "DNN Face Detection",
            "Thresholds Personalizados",
            "Métricas em Tempo Real"
        ],
        "pessoas_cadastradas": len(classifier.list_people()),
        "modelo_treinado": classifier.is_trained,
        "performance": performance_metrics
    }

@app.get("/api/status")
def get_status():
    """Status atual (para Pico W e Dashboard)"""
    return system_status.to_dict()

@app.get("/api/metrics")
def get_metrics():
    """Métricas de performance"""
    return {
        **performance_metrics,
        "model_stats": classifier.get_model_stats()
    }

@app.post("/api/registrar")
async def registrar(nome: str, files: List[UploadFile] = File(...)):
    """
    Registra pessoa com processamento profissional
    Gera múltiplos embeddings por imagem (augmentation)
    """
    if len(files) < 3:
        raise HTTPException(400, "Mínimo 3 imagens (recomendado: 5+)")
    
    print(f"\n{'='*60}")
    print(f"📝 REGISTRO: {nome}")
    print(f"{'='*60}")
    
    face_images = []
    
    for i, file in enumerate(files):
        print(f"\nProcessando imagem {i+1}/{len(files)}...")
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(400, f"Imagem inválida: {file.filename}")
        
        # Detectar rosto
        faces = detector.detect_faces(img)
        
        if len(faces) == 0:
            raise HTTPException(400, f"Nenhum rosto em: {file.filename}")
        
        # Analisar qualidade
        quality = detector.analyze_quality(img, faces[0]['bbox'])
        
        print(f"  Qualidade: {quality['quality_score']}/100")
        
        if quality['quality_score'] < 60:
            print(f"  ⚠ Qualidade baixa: {', '.join(quality['motivos'])}")
        
        # Extrair rosto
        face = detector.extract_face(img, faces[0]['bbox'])
        face_images.append(face)
    
    # Registrar
    success = classifier.register_person(nome, face_images)
    
    if not success:
        raise HTTPException(500, "Erro ao treinar modelo")
    
    print(f"\n✅ '{nome}' registrado com sucesso!")
    print(f"{'='*60}\n")
    
    return {
        "success": True,
        "message": f"'{nome}' registrado com augmentação automática",
        "imagens_enviadas": len(files),
        "embeddings_gerados": len(face_images) * 5,  # Aproximado
        "total_pessoas": len(classifier.list_people()),
        "model_stats": classifier.get_model_stats()
    }

@app.post("/api/reconhecer")
async def reconhecer(file: UploadFile = File(...)):
    """
    Reconhecimento PROFISSIONAL com ensemble e métricas
    """
    start_time = time.time()
    
    # Ler imagem
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(400, "Imagem inválida")
    
    # Detectar rosto
    faces = detector.detect_faces(img)
    
    if len(faces) == 0:
        result = {
            "nome": "desconhecido",
            "confianca": 0.0,
            "motivos": ["nenhum rosto detectado"],
            "acesso": "NAO",
            "quality_score": 0,
            "ensemble_votes": {}
        }
    else:
        # Extrair rosto
        face = detector.extract_face(img, faces[0]['bbox'])
        
        # Analisar qualidade
        quality = detector.analyze_quality(img, faces[0]['bbox'])
        
        # Reconhecer (ENSEMBLE)
        result = classifier.recognize(face, MIN_CONFIDENCE)
        result['quality_score'] = quality['quality_score']
        result['motivos'].extend(quality['motivos'])
        
        # Decisão de acesso
        if result['nome'] != "desconhecido" and \
           result['confianca'] >= MIN_CONFIDENCE and \
           quality['is_good_quality']:
            result['acesso'] = "SIM"
        else:
            result['acesso'] = "NAO"
            
            if not quality['is_good_quality']:
                result['motivos'].append(f"qualidade insuficiente ({quality['quality_score']}/100)")
        
        # Desenhar detecção
        img_box = detector.draw_detections(img, faces)
        
        # Base64
        _, buffer = cv2.imencode('.jpg', img_box)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        result['imagem'] = img_b64
    
    # Calcular tempo
    elapsed_ms = (time.time() - start_time) * 1000
    result['tempo_processamento_ms'] = round(elapsed_ms, 2)
    
    # Atualizar métricas
    performance_metrics['total_reconhecimentos'] += 1
    performance_metrics['ultimo_tempo_ms'] = elapsed_ms
    
    # Média móvel
    alpha = 0.1
    performance_metrics['tempo_medio_ms'] = \
        alpha * elapsed_ms + (1 - alpha) * performance_metrics['tempo_medio_ms']
    
    # Atualizar status global
    system_status.acesso = result['acesso']
    system_status.nome = result['nome']
    system_status.confianca = result['confianca']
    system_status.motivos = result['motivos']
    system_status.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    system_status.imagem_base64 = result.get('imagem', '')
    
    # Log
    access_logger.log_access(
        result['nome'],
        result['confianca'],
        result['acesso'] == "SIM",
        result['motivos']
    )
    
    # Broadcast WebSocket
    await manager.broadcast({
        "type": "recognition",
        "data": system_status.to_dict(),
        "metrics": {
            "tempo_ms": elapsed_ms,
            "quality_score": result['quality_score'],
            "ensemble": result.get('ensemble_votes', {})
        }
    })
    
    return result

@app.get("/api/pessoas")
def listar_pessoas():
    """Lista pessoas com estatísticas detalhadas"""
    return {
        "pessoas": classifier.list_people(),
        "total": len(classifier.list_people()),
        "model_stats": classifier.get_model_stats()
    }

@app.delete("/api/pessoa/{nome}")
def deletar_pessoa(nome: str):
    """Remove pessoa e retreina modelo"""
    success = classifier.delete_person(nome)
    if success:
        return {
            "success": True,
            "message": f"'{nome}' removido e modelo retreinado"
        }
    raise HTTPException(404, f"'{nome}' não encontrado")

@app.get("/api/historico")
def get_historico(limit: int = 100):
    """Histórico com análise estatística"""
    history = access_logger.get_history(limit)
    
    # Calcular estatísticas
    if history:
        liberados = sum(1 for h in history if h['acesso'] == 'LIBERADO')
        negados = sum(1 for h in history if h['acesso'] == 'NEGADO')
        confianca_media = np.mean([h['confianca'] for h in history])
        
        stats = {
            "total": len(history),
            "liberados": liberados,
            "negados": negados,
            "taxa_aprovacao": round(liberados / len(history) * 100, 1),
            "confianca_media": round(confianca_media, 3)
        }
    else:
        stats = {
            "total": 0,
            "liberados": 0,
            "negados": 0,
            "taxa_aprovacao": 0,
            "confianca_media": 0
        }
    
    return {
        "historico": history,
        "estatisticas": stats
    }

@app.get("/api/exportar")
def exportar_csv():
    """Exporta histórico para CSV"""
    csv_file = access_logger.export_csv()
    return FileResponse(csv_file, media_type='text/csv', filename=csv_file.name)

@app.get("/api/calibrar")
def calibrar_sistema():
    """
    Recalibra thresholds do sistema
    Útil após adicionar várias pessoas
    """
    if not classifier.is_trained:
        raise HTTPException(400, "Sistema não treinado")
    
    success = classifier.train()
    
    if success:
        return {
            "success": True,
            "message": "Sistema recalibrado com sucesso",
            "model_stats": classifier.get_model_stats()
        }
    
    raise HTTPException(500, "Erro ao recalibrar")

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket para atualizações em tempo real"""
    await manager.connect(websocket)
    
    try:
        # Status inicial
        await websocket.send_json({
            "type": "initial",
            "data": system_status.to_dict(),
            "metrics": performance_metrics
        })
        
        # Manter conexão
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Responder a pings
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "ping",
                    "metrics": performance_metrics
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.on_event("startup")
def startup():
    """Ao iniciar a API"""
    print("\n" + "="*70)
    print("🚀 SISTEMA DE RECONHECIMENTO FACIAL PROFISSIONAL v2.0")
    print("="*70)
    print(f"🔬 Detector: {type(detector).__name__}")
    print(f"🧠 Embeddings: Face Recognition (dlib) - 128D")
    print(f"🎯 Classificador: Ensemble (SVM + KNN + Distance)")
    print(f"📊 Pessoas: {len(classifier.list_people())}")
    print(f"📝 Histórico: {len(access_logger.get_history())} registros")
    
    if classifier.is_trained:
        stats = classifier.get_model_stats()
        print(f"✅ Modelo treinado: {stats['num_pessoas']} classes")
    else:
        print(f"⚠️  Modelo não treinado - registre pessoas")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    import uvicorn
    from ..utils.config import API_HOST, API_PORT
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)