import cv2
import os
import numpy as np
from tkinter import simpledialog, messagebox
from tkinter import Tk
import urllib.request


# --- Versão simplificada e funcional de reconhecimento facial ---
def get_face_cascade():
    # Tenta baixar o arquivo se necessário
    def download_file(url, filename):
        if not os.path.exists(filename):
            try:
                print(f"Baixando {filename}...")
                urllib.request.urlretrieve(url, filename)
                print("Download completo!")
                return True
            except Exception as e:
                print(f"Erro ao baixar {filename}: {e}")
                return False
        return True
    
    # URLs e caminhos possíveis
    face_xml_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    face_xml_path = 'haarcascade_frontalface_default.xml'
    
    # Baixa o arquivo se necessário
    if not download_file(face_xml_url, face_xml_path):
        raise FileNotFoundError("Não foi possível obter o classificador Haar Cascade")
    
    # Carrega o classificador
    cascade = cv2.CascadeClassifier(face_xml_path)
    
    if cascade.empty():
        raise FileNotFoundError("Classificador Haar Cascade carregado vazio")
    
    return cascade

# --- Pré-processamento da imagem ---
def preprocess_image(img):
    # Se a imagem for colorida, converte para cinza
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Equalização de histograma para melhor contraste
    gray = cv2.equalizeHist(gray)
    
    # Filtro Gaussiano para reduzir ruído
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return gray

# --- Função para comparar rostos ---
def compare_faces(face_img, known_faces, threshold=30):
    best_match = None
    min_diff = float('inf')
    
    # Pré-processa a imagem do rosto
    face_img = cv2.resize(face_img, (100, 100))
    face_img = cv2.equalizeHist(face_img)
    
    for name, known_face in known_faces.items():
        try:
            # Pré-processa o rosto conhecido
            known_face_proc = cv2.resize(known_face, (100, 100))
            known_face_proc = cv2.equalizeHist(known_face_proc)
            
            # Calcula a diferença entre as imagens
            diff = np.sum((face_img.astype("float") - known_face_proc.astype("float")) ** 2)
            diff /= float(face_img.shape[0] * face_img.shape[1])
            
            if diff < min_diff:
                min_diff = diff
                best_match = name
        except:
            continue
    
    # Debug: mostra a menor diferença encontrada
    print(f"Diferença mínima: {min_diff:.2f}, Reconhecido: {best_match}")
    return best_match if min_diff < threshold else None

# --- Carrega rostos conhecidos ---
def load_known_faces():
    known_faces = {}
    for filename in os.listdir(faces_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(faces_dir, filename)
            name = os.path.splitext(filename)[0]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                known_faces[name] = img
    print(f"Carregados {len(known_faces)} rostos conhecidos")
    return known_faces

# --- Configuração inicial ---
try:
    face_cascade = get_face_cascade()
except Exception as e:
    print(f"Erro fatal: {e}")
    exit()

faces_dir = 'rostos_salvos'
if not os.path.exists(faces_dir):
    os.makedirs(faces_dir)
    print(f"Diretório '{faces_dir}' criado")

known_faces = load_known_faces()
face_timers = {}
recognized_names = {}  # Dicionário para armazenar nomes reconhecidos

# --- Configuração da webcam ---
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Não foi possível abrir a câmera!")
    exit()

root = Tk()
root.withdraw()

frame_count = 0
last_recognition = {}  # Armazena o último resultado de reconhecimento por posição

print("Instruções:")
print("- Pressione 's' quando um rosto estiver visível para salvar")
print("- Pressione 'q' para sair")
print("- Para reconhecimento, apenas aponte a câmera para o rosto")

while True:
    ret, frame = video.read()
    if not ret:
        print("Erro ao capturar frame da câmera")
        break

    # Pré-processamento do frame
    gray = preprocess_image(frame)
    
    # Detecção de rostos
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(100, 100)
    )

    # Atualiza os temporizadores
    for face_id in list(face_timers.keys()):
        face_timers[face_id] -= 1
        if face_timers[face_id] <= 0:
            del face_timers[face_id]

    # Processa cada rosto detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        
        # Usa a posição como ID temporário
        face_id = f"{x}_{y}"
        
        # Verificação a cada 30 frames
        if frame_count % 30 == 0:
            name = compare_faces(roi_gray, known_faces)
            
            if name:
                # Armazena o nome reconhecido para exibição
                recognized_names[face_id] = name
                if face_id in face_timers:
                    del face_timers[face_id]
            elif face_id in recognized_names:
                # Remove o nome se não for mais reconhecido
                del recognized_names[face_id]
                face_timers[face_id] = 100  # Não perguntar novamente por 100 frames
            elif face_id not in face_timers:
                # Marca como visto para não perguntar novamente logo
                face_timers[face_id] = 100
        
        # Exibe o nome reconhecido acima do quadrado verde
        if face_id in recognized_names:
            name = recognized_names[face_id]
            # Posiciona o texto 10 pixels acima do retângulo
            text_y = y - 10 if y - 10 > 10 else y + 20
            cv2.putText(frame, name, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostra o frame
    cv2.imshow('Reconhecimento Facial (s=salvar, q=sair)', frame)
    frame_count += 1
    
    # Verifica teclas pressionadas
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Salva um novo rosto
        if len(faces) > 0:
            # Usa o primeiro rosto detectado
            x, y, w, h = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            
            name = simpledialog.askstring("Novo Rosto", "Digite o nome da pessoa:", parent=root)
            if name:
                face_filename = os.path.join(faces_dir, f"{name}.jpg")
                cv2.imwrite(face_filename, roi_gray)
                messagebox.showinfo("Sucesso", f"Rosto de {name} salvo com sucesso!", parent=root)
                known_faces = load_known_faces()  # Recarrega os rostos conhecidos
                print(f"Rosto de '{name}' salvo e recarregado")

# Limpeza
video.release()
cv2.destroyAllWindows()