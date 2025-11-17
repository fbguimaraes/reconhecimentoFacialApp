"""
Sistema de logs
"""
import json
from datetime import datetime
from pathlib import Path
from .config import LOGS_DIR

class AccessLogger:
    def __init__(self):
        hoje = datetime.now().strftime('%Y%m%d')
        self.json_file = LOGS_DIR / f"access_{hoje}.json"
        self.history = []
        self._load()
    
    def _load(self):
        """Carrega histórico do dia"""
        if self.json_file.exists():
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except:
                self.history = []
    
    def _save(self):
        """Salva histórico"""
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def log_access(self, nome, confianca, acesso, motivos):
        """Registra acesso"""
        entry = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "nome": nome,
            "confianca": round(confianca, 4),
            "acesso": "LIBERADO" if acesso else "NEGADO",
            "motivos": motivos
        }
        
        self.history.append(entry)
        self._save()
        
        print(f"[LOG] {entry['acesso']} - {nome} ({confianca:.1%})")
        return entry
    
    def get_history(self, limit=100):
        """Retorna últimos registros"""
        return self.history[-limit:]
    
    def export_csv(self):
        """Exporta CSV"""
        csv_file = LOGS_DIR / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("timestamp,nome,confianca,acesso,motivos\n")
            for entry in self.history:
                motivos = ';'.join(entry['motivos'])
                f.write(f"{entry['timestamp']},{entry['nome']},{entry['confianca']},{entry['acesso']},{motivos}\n")
        
        return csv_file

# Instância global
access_logger = AccessLogger()