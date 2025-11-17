"""
API FastAPI + WebSocket
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List
import cv2
import numpy as np
import base64
from datetime import datetime
import asyncio

from ..face.detector import FaceDetector
from ..face.classifier import FaceClassifier
from ..utils.logger import access_logger
from ..utils.config import system_status, MIN_CONFIDENCE

# Criar app
app = FastAPI(title="Sistema de Reconhecimento Facial")

# CORS (permitir acesso do dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar detector e classificador
detector = FaceDetector()
classifier = FaceClassifier()

# Gerenciador WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✓ WebSocket conectado ({len(self.active_connections)} total)")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"✗ WebSocket desconectado ({len(self.active_connections)} total)")
    
    async def broadcast(self, message: dict):
        """Envia para todos os clientes"""
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except:
                dead.append(conn)
        
        for conn in dead:
            self.active_connections.remove(conn)

manager = ConnectionManager()

# ROTAS

@app.get("/")
def root():
    """Endpoint raiz"""
    return {
        "message": "API de Reconhecimento Facial",
        "version": "1.0",
        "status": "online",
        "pessoas_cadastradas": len(classifier.list_people())
    }

@app.get("/api/status")
def get_status():
    """Status atual (para Pico W)"""
    return system_status.to_dict()

@app.post("/api/registrar")
async def registrar(nome: str, files: List[UploadFile] = File(...)):
    """
    Registra nova pessoa
    
    Args:
        nome: nome da pessoa
        files: 3+ imagens
    """
    if len(files) < 3:
        raise HTTPException(400, "Envie pelo menos 3 imagens")
    
    face_images = []
    
    for file in files:
        # Ler imagem
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(400, f"Imagem inválida: {file.filename}")
        
        # Detectar rosto
        faces = detector.detect_faces(img)
        
        if len(faces) == 0:
            raise HTTPException(400, f"Nenhum rosto em: {file.filename}")
        
        # Extrair rosto
        face = detector.extract_face(img, faces[0]['bbox'])
        face_images.append(face)
    
    # Registrar
    success = classifier.register_person(nome, face_images)
    
    if not success:
        raise HTTPException(500, "Erro ao treinar modelo")
    
    return {
        "success": True,
        "message": f"'{nome}' registrado com {len(face_images)} imagens",
        "total_pessoas": len(classifier.list_people())
    }

@app.post("/api/reconhecer")
async def reconhecer(file: UploadFile = File(...)):
    """
    Reconhece pessoa na imagem
    """
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
            "acesso": "NAO"
        }
    else:
        # Extrair rosto
        face = detector.extract_face(img, faces[0]['bbox'])
        
        # Analisar qualidade
        quality = detector.analyze_quality(img, faces[0]['bbox'])
        
        # Reconhecer
        result = classifier.recognize(face, MIN_CONFIDENCE)
        result['motivos'].extend(quality['motivos'])
        
        # Acesso
        if result['nome'] != "desconhecido" and result['confianca'] >= MIN_CONFIDENCE:
            result['acesso'] = "SIM"
        else:
            result['acesso'] = "NAO"
        
        # Desenhar detecção
        img_box = detector.draw_detections(img, faces)
        
        # Base64
        _, buffer = cv2.imencode('.jpg', img_box)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        result['imagem'] = img_b64
    
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
        "data": system_status.to_dict()
    })
    
    return result

@app.get("/api/pessoas")
def listar_pessoas():
    """Lista pessoas cadastradas"""
    return classifier.list_people()

@app.delete("/api/pessoa/{nome}")
def deletar_pessoa(nome: str):
    """Remove pessoa"""
    success = classifier.delete_person(nome)
    if success:
        return {"success": True, "message": f"'{nome}' removido"}
    raise HTTPException(404, f"'{nome}' não encontrado")

@app.get("/api/historico")
def get_historico(limit: int = 100):
    """Histórico de acessos"""
    return access_logger.get_history(limit)

@app.get("/api/exportar")
def exportar_csv():
    """Exporta CSV"""
    csv_file = access_logger.export_csv()
    return FileResponse(csv_file, media_type='text/csv', filename=csv_file.name)

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket para tempo real"""
    await manager.connect(websocket)
    
    try:
        # Enviar status inicial
        await websocket.send_json({
            "type": "initial",
            "data": system_status.to_dict()
        })
        
        # Manter conexão
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Erro WebSocket: {e}")
        manager.disconnect(websocket)

@app.on_event("startup")
def startup():
    """Ao iniciar"""
    print("\n" + "="*60)
    print("🚀 API de Reconhecimento Facial")
    print("="*60)
    print(f"📊 Pessoas: {len(classifier.list_people())}")
    print(f"📝 Histórico: {len(access_logger.get_history())} registros")
    print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    from ..utils.config import API_HOST, API_PORT
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)