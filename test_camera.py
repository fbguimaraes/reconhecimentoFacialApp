"""
Script de Teste do Sistema de Reconhecimento Facial
Facilita registro e testes
"""
import cv2
import requests
import sys
import os

API_URL = "http://localhost:8000/api"

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

def menu():
    limpar_tela()
    print("="*60)
    print("  SISTEMA DE RECONHECIMENTO FACIAL - TESTE")
    print("="*60)
    print()
    print("1. Registrar Nova Pessoa (capturar 3 fotos)")
    print("2. Testar Reconhecimento (foto única)")
    print("3. Testar Reconhecimento (contínuo)")
    print("4. Listar Pessoas Cadastradas")
    print("5. Ver Status Atual")
    print("0. Sair")
    print()

def capturar_fotos(num=3):
    """Captura fotos da webcam"""
    print(f"\n📸 Capturando {num} fotos...")
    print("Instruções:")
    print("  - Pressione ESPAÇO para capturar cada foto")
    print("  - Pressione ESC para cancelar")
    print()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erro: Webcam não encontrada!")
        return None
    
    fotos = []
    count = 0
    
    while count < num:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erro ao capturar frame")
            break
        
        # Mostrar preview
        display = frame.copy()
        cv2.putText(display, f"Foto {count + 1}/{num} - ESPACO: Capturar | ESC: Cancelar", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Captura de Fotos', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # ESPAÇO
            fotos.append(frame.copy())
            count += 1
            print(f"✓ Foto {count}/{num} capturada!")
        elif key == 27:  # ESC
            print("❌ Cancelado")
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✓ {len(fotos)} fotos capturadas com sucesso!")
    return fotos

def registrar_pessoa():
    """Registra nova pessoa"""
    print("\n" + "="*60)
    print("  REGISTRAR NOVA PESSOA")
    print("="*60)
    
    nome = input("\nDigite o nome da pessoa: ").strip()
    
    if not nome:
        print("❌ Nome não pode ser vazio")
        input("\nPressione ENTER...")
        return
    
    print(f"\nPreparando para capturar fotos de: {nome}")
    print("Dicas:")
    print("  - Foto 1: Olhando para frente")
    print("  - Foto 2: Virado levemente para esquerda")
    print("  - Foto 3: Virado levemente para direita")
    print()
    input("Pressione ENTER para começar...")
    
    fotos = capturar_fotos(3)
    
    if not fotos:
        input("\nPressione ENTER...")
        return
    
    print("\n📤 Enviando para a API...")
    
    files = []
    for i, foto in enumerate(fotos):
        _, buffer = cv2.imencode('.jpg', foto)
        files.append(('files', (f'foto{i}.jpg', buffer.tobytes(), 'image/jpeg')))
    
    try:
        response = requests.post(
            f"{API_URL}/registrar",
            params={'nome': nome},
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCESSO!")
            print(f"   Nome: {nome}")
            print(f"   Fotos registradas: {len(fotos)}")
            print(f"   Total de pessoas: {result.get('total_pessoas', '?')}")
        else:
            print(f"\n❌ Erro: {response.status_code}")
            print(f"   {response.text}")
    
    except Exception as e:
        print(f"\n❌ Erro de conexão: {e}")
        print("   Verifique se a API está rodando!")
    
    input("\nPressione ENTER...")

def testar_unico():
    """Testa reconhecimento com uma foto"""
    print("\n" + "="*60)
    print("  TESTE DE RECONHECIMENTO (FOTO ÚNICA)")
    print("="*60)
    print("\nPressione ESPAÇO para capturar foto")
    input("Pressione ENTER para começar...")
    
    fotos = capturar_fotos(1)
    
    if not fotos:
        input("\nPressione ENTER...")
        return
    
    print("\n📤 Enviando para reconhecimento...")
    
    _, buffer = cv2.imencode('.jpg', fotos[0])
    
    try:
        response = requests.post(
            f"{API_URL}/reconhecer",
            files={'file': ('foto.jpg', buffer.tobytes(), 'image/jpeg')},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n" + "="*60)
            print("  RESULTADO")
            print("="*60)
            print(f"Nome: {result['nome']}")
            print(f"Confiança: {result['confianca']*100:.1f}%")
            print(f"Acesso: {result['acesso']}")
            
            if result['motivos']:
                print("\nMotivos:")
                for motivo in result['motivos']:
                    print(f"  • {motivo}")
        else:
            print(f"\n❌ Erro: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"\n❌ Erro: {e}")
    
    input("\nPressione ENTER...")

def testar_continuo():
    """Testa reconhecimento contínuo"""
    print("\n" + "="*60)
    print("  TESTE CONTÍNUO")
    print("="*60)
    print("\nProcessando frames a cada 1 segundo")
    print("Pressione ESC para parar")
    input("\nPressione ENTER para começar...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Webcam não encontrada")
        input("\nPressione ENTER...")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Processar a cada 30 frames (~1 seg)
        if frame_count % 30 == 0:
            _, buffer = cv2.imencode('.jpg', frame)
            
            try:
                response = requests.post(
                    f"{API_URL}/reconhecer",
                    files={'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')},
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Mostrar no frame
                    nome = result['nome']
                    conf = result['confianca'] * 100
                    acesso = result['acesso']
                    
                    color = (0, 255, 0) if acesso == "SIM" else (0, 0, 255)
                    
                    cv2.putText(frame, f"Nome: {nome}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"Confianca: {conf:.1f}%", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"Acesso: {acesso}", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    print(f"✓ {nome} | {conf:.1f}% | {acesso}")
            
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        cv2.putText(frame, "ESC: Sair", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Reconhecimento Continuo', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    input("\nPressione ENTER...")

def listar_pessoas():
    """Lista pessoas cadastradas"""
    print("\n" + "="*60)
    print("  PESSOAS CADASTRADAS")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/pessoas", timeout=5)
        
        if response.status_code == 200:
            pessoas = response.json()
            
            if not pessoas:
                print("\n⚠ Nenhuma pessoa cadastrada ainda")
            else:
                print(f"\nTotal: {len(pessoas)} pessoas\n")
                for p in pessoas:
                    print(f"  • {p['nome']} ({p['num_embeddings']} fotos)")
        else:
            print(f"\n❌ Erro: {response.status_code}")
    
    except Exception as e:
        print(f"\n❌ Erro de conexão: {e}")
    
    input("\nPressione ENTER...")

def ver_status():
    """Mostra status atual"""
    print("\n" + "="*60)
    print("  STATUS DO SISTEMA")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        
        if response.status_code == 200:
            status = response.json()
            
            print(f"\nAcesso: {status['acesso']}")
            print(f"Nome: {status['nome']}")
            print(f"Confiança: {status['confianca']*100:.1f}%")
            print(f"Timestamp: {status['timestamp']}")
            
            if status['motivos']:
                print("\nMotivos:")
                for m in status['motivos']:
                    print(f"  • {m}")
        else:
            print(f"\n❌ Erro: {response.status_code}")
    
    except Exception as e:
        print(f"\n❌ Erro de conexão: {e}")
        print("Verifique se a API está rodando!")
    
    input("\nPressione ENTER...")

def main():
    """Loop principal"""
    while True:
        menu()
        escolha = input("Escolha uma opção: ").strip()
        
        if escolha == "1":
            registrar_pessoa()
        elif escolha == "2":
            testar_unico()
        elif escolha == "3":
            testar_continuo()
        elif escolha == "4":
            listar_pessoas()
        elif escolha == "5":
            ver_status()
        elif escolha == "0":
            print("\n👋 Até logo!")
            sys.exit(0)
        else:
            print("\n❌ Opção inválida")
            input("Pressione ENTER...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrompido")
        sys.exit(0)