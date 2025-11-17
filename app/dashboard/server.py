"""
Servidor do Dashboard Web
"""
from flask import Flask, render_template, send_from_directory
from pathlib import Path

# Diretórios
template_dir = Path(__file__).parent / 'templates'
static_dir = Path(__file__).parent / 'static'

app = Flask(
    __name__,
    template_folder=str(template_dir),
    static_folder=str(static_dir)
)

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/historico')
def historico():
    """Página de histórico"""
    return render_template('historico.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve arquivos estáticos"""
    return send_from_directory(str(static_dir), filename)

if __name__ == '__main__':
    from ..utils.config import DASHBOARD_HOST, DASHBOARD_PORT
    
    print("\n" + "="*60)
    print("🎨 Dashboard Web")
    print(f"📍 http://localhost:{DASHBOARD_PORT}")
    print("="*60 + "\n")
    
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False)