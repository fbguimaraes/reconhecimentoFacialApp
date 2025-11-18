# 🚀 INSTALAÇÃO EM OUTRO PC

## 📋 Pré-requisitos

- Python 3.11 (mesma versão)
- Git instalado
- Webcam conectada

---

## 🔧 PASSO A PASSO

### 1️⃣ Clonar repositório

```bash
git clone https://github.com/SEU_USUARIO/sistema_facial.git
cd sistema_facial
```

### 2️⃣ Criar ambiente virtual

```bash
python -m venv venv
```

**Ativar:**

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

### 3️⃣ Atualizar pip

```bash
python -m pip install --upgrade pip
```

### 4️⃣ Instalar dependências básicas

```bash
pip install fastapi uvicorn[standard] websockets python-multipart
pip install mtcnn opencv-python opencv-contrib-python
pip install scikit-learn numpy scipy pillow pydantic python-dotenv aiofiles flask
```

### 5️⃣ Instalar dlib pré-compilado

**⚠️ IMPORTANTE: Passo específico para dlib**

```bash
pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp311-cp311-win_amd64.whl
pip install face-recognition
```

**Se der erro:**
- Verificar se é Python 3.11
- Baixar manualmente o arquivo .whl
- Instalar: `pip install caminho/para/arquivo.whl`

**Se ainda não funcionar:**
Use a versão sem dlib (ver seção "Alternativa" no final)

---

## ✅ Verificar instalação

```bash
python -c "import face_recognition; print('OK')"
```

Se aparecer "OK", está pronto!

---

## 🎮 Iniciar sistema

**Terminal 1 - API:**
```bash
python -m app.api.main
```

**Terminal 2 - Dashboard:**
```bash
python -m app.dashboard.server
```

**Terminal 3 - Teste:**
```bash
python test_camera.py
```

---

## 🔄 ALTERNATIVA: Sem dlib (fallback)

Se dlib não instalar de jeito nenhum:

1. **Baixar arquivo alternativo:**
   - Vá em: [link do seu GitHub]/blob/main/app/face/embeddings_no_dlib.py
   - Salve como `embeddings.py`

2. **Substituir:**
   ```bash
   copy embeddings_no_dlib.py app\face\embeddings.py
   ```

3. **Não instalar dlib:**
   ```bash
   # Pular etapa 5, instalar só isso:
   pip install mtcnn opencv-python opencv-contrib-python
   ```

Sistema funcionará com 85-90% da precisão (ainda muito bom).

---

## 🐛 Troubleshooting

### Erro: "No module named 'cv2'"
```bash
pip install opencv-python
```

### Erro: "CMake not found"
Use o dlib pré-compilado (etapa 5) ou versão sem dlib.

### Erro: "Face_recognition not found"
```bash
pip install face-recognition
```

### Porta já em uso
Edite `app/utils/config.py` e mude as portas.
