# 🔐 Sistema de Reconhecimento Facial + Controle de Acesso

Sistema completo de reconhecimento facial com dashboard web em tempo real e integração com Raspberry Pi Pico W para controle físico de acesso.

---

## 📋 ÍNDICE

1. [Visão Geral](#visão-geral)
2. [Funcionalidades](#funcionalidades)
3. [Requisitos](#requisitos)
4. [Instalação](#instalação)
5. [Como Usar](#como-usar)
6. [Estrutura do Projeto](#estrutura-do-projeto)
7. [API Endpoints](#api-endpoints)
8. [Integração com Pico W](#integração-com-pico-w)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 VISÃO GERAL

Este sistema permite:

- **Reconhecimento facial** em tempo real usando webcam
- **Dashboard web** com visualização ao vivo dos acessos
- **API REST** completa para integração
- **WebSocket** para atualizações instantâneas
- **Logs automáticos** de todos os acessos
- **Integração com hardware** (Raspberry Pi Pico W + LEDs + OLED)

### Arquitetura

```
┌─────────────────────────────────────────────┐
│           SISTEMA COMPLETO                  │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────┐   ┌────────────┐            │
│  │ WEBCAM   │──▶│  PYTHON    │            │
│  │          │   │  (FastAPI) │            │
│  └──────────┘   └─────┬──────┘            │
│                       │                     │
│                       │ WebSocket           │
│                       ▼                     │
│                ┌─────────────┐             │
│                │  DASHBOARD  │             │
│                │   (Flask)   │             │
│                └─────────────┘             │
│                       │                     │
│                       │ HTTP                │
│                       ▼                     │
│                ┌─────────────┐             │
│                │   PICO W    │             │
│                │ LED + OLED  │             │
│                └─────────────┘             │
│                                             │
└─────────────────────────────────────────────┘
```

---

## ✨ FUNCIONALIDADES

### Reconhecimento Facial
- ✅ Detecção com Haar Cascade (OpenCV)
- ✅ Geração de embeddings (HOG + histograma de cores)
- ✅ Classificação SVM (poucas amostras necessárias)
- ✅ Análise de qualidade da imagem
- ✅ Threshold de confiança configurável

### Dashboard Web
- ✅ Visualização em tempo real
- ✅ LED virtual (verde/vermelho/cinza)
- ✅ Exibição da última foto capturada
- ✅ Estatísticas de confiança e motivos
- ✅ Histórico completo de acessos
- ✅ Filtros por nome e status
- ✅ Exportação para CSV

### API REST
- ✅ Registro de pessoas (múltiplas fotos)
- ✅ Reconhecimento facial
- ✅ Listagem de pessoas cadastradas
- ✅ Exclusão de pessoas
- ✅ Histórico de acessos
- ✅ Status atual do sistema
- ✅ WebSocket para tempo real

### Hardware (Pico W)
- ✅ LED verde (acesso liberado)
- ✅ LED vermelho (acesso negado)
- ✅ Display OLED I2C (nome e confiança)
- ✅ Conexão WiFi
- ✅ Consulta HTTP à API

---

## 📦 REQUISITOS

### Software

- **Python 3.8 a 3.11** (não use 3.12+)
- **Webcam** (integrada ou USB)
- **Windows 10/11** (ou Linux/Mac com adaptações)

### Hardware (Opcional - Pico W)

- Raspberry Pi Pico W
- LED Verde + Resistor 220Ω
- LED Vermelho + Resistor 220Ω
- Display OLED SSD1306 128x64 (I2C)
- Protoboard e jumpers

---

## 🚀 INSTALAÇÃO

### Passo 1: Clonar/Criar Estrutura

```powershell
# Criar pasta do projeto
mkdir C:\Users\SEU_USUARIO\sistema_facial
cd C:\Users\SEU_USUARIO\sistema_facial

# Copiar todos os arquivos do sistema para esta pasta
```

Estrutura final:

```
sistema_facial/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── face/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   ├── embeddings.py
│   │   ├── classifier.py
│   │   └── data/          # Embeddings salvos aqui
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── templates/
│   │   │   ├── index.html
│   │   │   └── historico.html
│   │   └── static/        # CSS/JS (vazio por enquanto)
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
├── logs/                  # Logs automáticos
├── pico/                  # Código do Pico W (futuro)
├── requirements.txt
├── test_camera.py
└── README.md
```

### Passo 2: Criar Ambiente Virtual

```powershell
# Criar ambiente virtual
python -m venv venv

# Ativar (SEMPRE ativar antes de usar)
venv\Scripts\activate

# Você verá (venv) no início da linha
```

### Passo 3: Instalar Dependências

```powershell
# Atualizar pip
python -m pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt

# Aguardar instalação (2-5 minutos)
```

### Passo 4: Verificar Instalação

```powershell
# Testar importações
python -c "from app.face.detector import FaceDetector; print('OK')"

# Deve mostrar:
# ✓ Detector de rostos inicializado
# ✓ Gerador de embeddings inicializado
# OK
```

---

## 🎮 COMO USAR

### Inicialização do Sistema (3 Terminais)

O sistema precisa de **3 terminais simultâneos**:

#### **TERMINAL 1: API**

```powershell
# Abrir PowerShell
cd C:\Users\SEU_USUARIO\sistema_facial
venv\Scripts\activate
python -m app.api.main
```

**Saída esperada:**
```
✓ Detector de rostos inicializado
✓ Gerador de embeddings inicializado
======================================================
🚀 API de Reconhecimento Facial
======================================================
📊 Pessoas: 0
📝 Histórico: 0 registros
======================================================

INFO: Uvicorn running on http://0.0.0.0:8000
```

**Deixe rodando!**

---

#### **TERMINAL 2: Dashboard**

```powershell
# Abrir NOVO PowerShell
cd C:\Users\SEU_USUARIO\sistema_facial
venv\Scripts\activate
python -m app.dashboard.server
```

**Saída esperada:**
```
======================================================
🎨 Dashboard Web
📍 http://localhost:5000
======================================================

 * Running on http://0.0.0.0:5000
```

**Deixe rodando!**

Abra no navegador: **http://localhost:5000**

---

#### **TERMINAL 3: Script de Teste**

```powershell
# Abrir NOVO PowerShell
cd C:\Users\SEU_USUARIO\sistema_facial
venv\Scripts\activate
python test_camera.py
```

**Você verá o menu interativo!**

---

### Usar o Sistema

#### **1. Registrar Primeira Pessoa**

No Terminal 3 (script de teste):

```
1. Registrar Nova Pessoa (capturar 3 fotos)
```

1. Digite `1` e ENTER
2. Digite o nome (ex: "Fabricio")
3. ENTER para começar
4. **Janela da webcam abrirá**
5. Posicione seu rosto na frente da câmera
6. Pressione **ESPAÇO** 3 vezes:
   - Foto 1: Olhando de frente
   - Foto 2: Cabeça virada 15° à esquerda
   - Foto 3: Cabeça virada 15° à direita
7. Aguarde upload

**Mensagem de sucesso:**
```
✅ SUCESSO!
   Nome: Fabricio
   Fotos registradas: 3
   Total de pessoas: 1
```

---

#### **2. Testar Reconhecimento**

**Opção A: Foto Única**

No menu, digite `2`:

1. ENTER para começar
2. Posicione-se na frente da webcam
3. Pressione ESPAÇO
4. **Veja o resultado:**

```
======================================================
  RESULTADO
======================================================
Nome: Fabricio
Confiança: 87.3%
Acesso: SIM
```

**Opção B: Reconhecimento Contínuo**

No menu, digite `3`:

1. ENTER para começar
2. Fique na frente da webcam
3. O sistema reconhece automaticamente a cada 1 segundo
4. Veja na tela: nome, confiança e status
5. Pressione ESC para sair

---

#### **3. Ver no Dashboard**

Volte ao navegador: **http://localhost:5000**

**Você verá:**
- 🟢 LED VERDE (se reconheceu)
- ✅ "ACESSO LIBERADO"
- 👤 Seu nome
- 📊 Confiança (%)
- 🖼️ Sua foto com retângulo verde
- ✅ Status: LIBERADO

**Se não reconheceu:**
- 🔴 LED VERMELHO
- ❌ "ACESSO NEGADO"
- ⚠️ Motivos (confiança baixa, etc)

---

#### **4. Ver Histórico**

No dashboard, acesse: **http://localhost:5000/historico**

**Você verá:**
- Tabela com todos os acessos
- Data/hora
- Nome
- Confiança
- Status (LIBERADO/NEGADO)
- Motivos
- Botão para exportar CSV

---

### Cadastrar Mais Pessoas

Repita o processo no menu opção `1` com nomes diferentes:
- "Maria"
- "João"
- "Ana"
- etc.

**Mínimo recomendado:** 3 fotos por pessoa  
**Ideal:** 5+ fotos com iluminação variada

---

## 📁 ESTRUTURA DO PROJETO

### Arquivos Principais

| Arquivo | Função |
|---------|--------|
| `app/face/detector.py` | Detecta rostos com Haar Cascade |
| `app/face/embeddings.py` | Gera vetores de características |
| `app/face/classifier.py` | Classifica rostos com SVM |
| `app/api/main.py` | API REST + WebSocket |
| `app/dashboard/server.py` | Servidor web do dashboard |
| `app/utils/config.py` | Configurações globais |
| `app/utils/logger.py` | Sistema de logs |
| `test_camera.py` | Script de teste interativo |

### Diretórios de Dados

| Diretório | Conteúdo |
|-----------|----------|
| `app/face/data/` | Embeddings de cada pessoa |
| `app/face/data/NOME/` | Embeddings da pessoa NOME |
| `logs/` | Logs diários em JSON |

---

## 📡 API ENDPOINTS

### Status do Sistema

```http
GET http://localhost:8000/api/status
```

**Resposta:**
```json
{
  "acesso": "SIM",
  "nome": "Fabricio",
  "confianca": 0.873,
  "motivos": [],
  "timestamp": "2024-11-15 14:30:45",
  "imagem": "base64_string..."
}
```

---

### Registrar Pessoa

```http
POST http://localhost:8000/api/registrar?nome=Fabricio
Content-Type: multipart/form-data

files: [imagem1.jpg, imagem2.jpg, imagem3.jpg]
```

**Resposta:**
```json
{
  "success": true,
  "message": "'Fabricio' registrado com 3 imagens",
  "total_pessoas": 5
}
```

---

### Reconhecer Pessoa

```http
POST http://localhost:8000/api/reconhecer
Content-Type: multipart/form-data

file: imagem.jpg
```

**Resposta:**
```json
{
  "nome": "Fabricio",
  "confianca": 0.873,
  "acesso": "SIM",
  "motivos": [],
  "embedding_dist": 0.42,
  "imagem": "base64_com_bbox..."
}
```

---

### Listar Pessoas

```http
GET http://localhost:8000/api/pessoas
```

**Resposta:**
```json
[
  {
    "nome": "Fabricio",
    "num_embeddings": 3
  },
  {
    "nome": "Maria",
    "num_embeddings": 5
  }
]
```

---

### Histórico

```http
GET http://localhost:8000/api/historico?limit=100
```

**Resposta:**
```json
[
  {
    "timestamp": "2024-11-15 14:30:45",
    "nome": "Fabricio",
    "confianca": 0.873,
    "acesso": "LIBERADO",
    "motivos": []
  }
]
```

---

### Exportar CSV

```http
GET http://localhost:8000/api/exportar
```

Retorna arquivo CSV para download.

---

### WebSocket

```javascript
ws://localhost:8000/ws/events
```

**Mensagens recebidas:**
```json
{
  "type": "recognition",
  "data": {
    "acesso": "SIM",
    "nome": "Fabricio",
    "confianca": 0.873,
    ...
  }
}
```

---

## 🔧 INTEGRAÇÃO COM PICO W

### Hardware Necessário

- Raspberry Pi Pico W
- LED Verde → GPIO 15 + Resistor 220Ω → GND
- LED Vermelho → GPIO 14 + Resistor 220Ω → GND
- OLED SSD1306 I2C:
  - SDA → GPIO 4
  - SCL → GPIO 5
  - VCC → 3V3
  - GND → GND

### Configuração

1. Descobrir IP do PC:

```powershell
ipconfig
# Anotar "Endereço IPv4" (ex: 192.168.1.100)
```

2. Editar `pico/main.c`:

```c
#define WIFI_SSID "SUA_REDE_WIFI"
#define WIFI_PASSWORD "SUA_SENHA"
#define API_HOST "192.168.1.100"  // IP do seu PC
```

3. Compilar e fazer upload (instruções detalhadas no código)

### Funcionamento

O Pico W:
1. Conecta ao WiFi
2. A cada 1 segundo, consulta: `GET /api/status`
3. Lê campo `"acesso"`
4. Se `"SIM"`: LED verde + OLED mostra nome e confiança
5. Se `"NAO"`: LED vermelho + OLED mostra "Acesso Negado"

---

## 🐛 TROUBLESHOOTING

### Erro: "ModuleNotFoundError: No module named 'app'"

**Solução:** Você está na pasta errada.

```powershell
cd C:\Users\SEU_USUARIO\sistema_facial
python -m app.api.main
```

---

### Erro: "Webcam não encontrada"

**Causas:**
- Webcam não conectada
- Outra aplicação usando a webcam
- Drivers desatualizados

**Solução:**
1. Conectar webcam USB
2. Fechar outras aplicações (Zoom, Teams, etc)
3. Testar com: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

---

### Erro: "No matching distribution found"

**Solução:** Sua versão do Python pode ser incompatível.

```powershell
python --version
# Deve ser 3.8 a 3.11

# Se for 3.12+, instale Python 3.11
```

---

### Dashboard não conecta

**Sintomas:** "Reconectando..." no dashboard

**Soluções:**
1. Verificar se API está rodando no Terminal 1
2. Verificar firewall não está bloqueando porta 8000
3. Limpar cache do navegador (Ctrl+F5)
4. Abrir console do navegador (F12) para ver erros

---

### Confiança sempre baixa

**Causas:**
- Poucas fotos de treinamento
- Fotos com iluminação diferente
- Ângulos muito variados

**Solução:**
1. Registrar 5+ fotos
2. Capturar em condições similares ao uso
3. Ajustar `MIN_CONFIDENCE` em `config.py` (ex: 0.60)

---

### "Nenhum rosto detectado"

**Causas:**
- Rosto muito pequeno na imagem
- Iluminação muito baixa/alta
- Ângulo extremo

**Solução:**
1. Aproximar rosto da câmera
2. Melhorar iluminação
3. Olhar de frente para a câmera

---

## 📊 CONFIGURAÇÕES AVANÇADAS

### Ajustar Confiança Mínima

Edite `app/utils/config.py`:

```python
MIN_CONFIDENCE = 0.70  # 70% (padrão)
# Valores:
# 0.60 = mais permissivo (mais falsos positivos)
# 0.80 = mais restritivo (mais falsos negativos)
```

Reinicie a API após alterar.

---

### Alterar Porta da API

```python
# app/utils/config.py
API_PORT = 8000  # Mudar se necessário
```

Atualizar também no dashboard e script de teste.

---

### Logs

Logs são salvos automaticamente em: `logs/access_YYYYMMDD.json`

Formato:
```json
{
  "timestamp": "2024-11-15 14:30:45",
  "nome": "Fabricio",
  "confianca": 0.873,
  "acesso": "LIBERADO",
  "motivos": []
}
```

---

## 🎓 COMO FUNCIONA

### 1. Detecção de Rostos

Usa **Haar Cascade** (OpenCV):
- Detecta faces em tempo real
- Retorna bounding box (x, y, largura, altura)
- Rápido e leve

### 2. Geração de Embeddings

Extrai características do rosto:
- **HOG (Histogram of Oriented Gradients)**: Gradientes da imagem
- **Histograma de cores**: Distribuição RGB
- **Estatísticas**: Médias e desvios

Resultado: vetor de 128 dimensões

### 3. Classificação SVM

Treina **Support Vector Machine**:
- Aprende padrões de cada pessoa
- Funciona com poucas amostras (few-shot)
- Retorna probabilidade para cada classe

### 4. Decisão Final

```python
if confianca >= MIN_CONFIDENCE and distancia <= threshold:
    acesso = "SIM"
else:
    acesso = "NAO"
```

---

## 🚀 PRÓXIMOS PASSOS

### Melhorias Possíveis

1. **Anti-Spoofing**: Detectar fotos/vídeos falsos
2. **Múltiplas Câmeras**: Vários pontos de acesso
3. **Banco de Dados**: PostgreSQL ao invés de arquivos
4. **Notificações**: Email/SMS em acessos negados
5. **App Mobile**: Dashboard nativo iOS/Android
6. **Face Mask Detection**: Detectar uso de máscara
7. **Relatórios**: Gerar PDFs automáticos

### Integração com BitDogLab

O sistema foi projetado para **fácil integração**:
- API REST padrão (HTTP/JSON)
- WebSocket para tempo real
- Logs estruturados
- Modular e extensível

---

## 📞 SUPORTE

### Antes de Pedir Ajuda

1. ✅ Ler este README completo
2. ✅ Verificar [Troubleshooting](#troubleshooting)
3. ✅ Consultar logs em `logs/`
4. ✅ Ver mensagens de erro completas

### Informações Úteis para Debug

```powershell
# Versão Python
python --version

# Pacotes instalados
pip list

# Testar webcam
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'ERRO')"

# Testar API
curl http://localhost:8000
```

---

## 📝 CHANGELOG

### v1.0.0 (2024-11-15)
- ✅ Sistema inicial completo
- ✅ Reconhecimento facial com Haar Cascade
- ✅ API REST + WebSocket
- ✅ Dashboard web em tempo real
- ✅ Sistema de logs
- ✅ Script de teste interativo
- ✅ Suporte ao Raspberry Pi Pico W

---

## 📄 LICENÇA

Este projeto é fornecido para fins educacionais.

---

## 🎉 CONCLUSÃO

Você agora tem um sistema completo de reconhecimento facial funcional!

**Recursos:**
- ✅ API profissional
- ✅ Dashboard moderno
- ✅ Reconhecimento em tempo real
- ✅ Logs detalhados
- ✅ Integração com hardware
- ✅ Código limpo e documentado

**Bom uso! 🚀**

---

**Desenvolvido para aprendizado de sistemas embarcados e visão computacional.**