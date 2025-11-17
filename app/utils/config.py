"""
Configurações do sistema
"""
import os
from pathlib import Path

# Diretórios
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "app" / "face" / "data"
LOGS_DIR = BASE_DIR / "logs"

# Criar se não existem
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configurações de reconhecimento
FACE_SIZE = (160, 160)
MIN_CONFIDENCE = 0.70  # 70%

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# Dashboard
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5000

# Webcam
CAMERA_INDEX = 0

# Status global do sistema
class SystemStatus:
    def __init__(self):
        self.acesso = "NAO"
        self.nome = "aguardando"
        self.confianca = 0.0
        self.motivos = []
        self.timestamp = ""
        self.imagem_base64 = ""
    
    def to_dict(self):
        return {
            "acesso": self.acesso,
            "nome": self.nome,
            "confianca": self.confianca,
            "motivos": self.motivos,
            "timestamp": self.timestamp,
            "imagem": self.imagem_base64
        }

system_status = SystemStatus()