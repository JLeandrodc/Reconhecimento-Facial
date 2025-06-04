import cv2
import os
import numpy as np
from tkinter import simpledialog, messagebox
from tkinter import Tk
import urllib.request
from datetime import datetime
import time

# --- Configurações iniciais ---
FACE_CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
EYE_CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml'

# --- Funções auxiliares ---
def download_file(url, filename):
    """Baixa um arquivo se não existir localmente"""
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

def get_face_cascade():
    """Carrega o classificador de faces"""
    if not download_file(FACE_CASCADE_URL, 'haarcascade_frontalface_default.xml'):
        raise FileNotFoundError("Não foi possível obter o classificador de faces")
    
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if cascade.empty():
        raise FileNotFoundError("Classificador de faces carregado vazio")
    return cascade

def get_eye_cascade():
    """Carrega o classificador de olhos"""
    if not download_file(EYE_CASCADE_URL, 'haarcascade_eye.xml'):
        raise FileNotFoundError("Não foi possível obter o classificador de olhos")
    
    cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    if cascade.empty():
        raise FileNotFoundError("Classificador de olhos carregado vazio")
    return cascade

def preprocess_image(img):
    """Pré-processa a imagem para melhor detecção"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def create_attendance_file():
    """Cria arquivo de presença com data atual"""
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"presenca_{today}.txt"
    
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(f"Registro de Presença - {today}\n")
            f.write("="*30 + "\n")
    
    return filename

def register_attendance(name, filename):
    """Registra a presença no arquivo"""
    now = datetime.now().strftime("%H:%M:%S")
    with open(filename, 'a') as f:
        f.write(f"{name} - {now}\n")

# --- Sistema de reconhecimento facial ---
class FaceRecognizer:
    def __init__(self):
        self.face_cascade = get_face_cascade()
        self.eye_cascade = get_eye_cascade()
        self.faces_dir = 'rostos_salvos'
        self.attendance_file = create_attendance_file()
        
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
        
        self.known_faces = self.load_known_faces()
        self.face_timers = {}
        self.recognized_names = {}
        self.last_seen = {}

    def load_known_faces(self):
        """Carrega rostos conhecidos com múltiplas poses"""
        known_faces = {}
        
        for person_dir in os.listdir(self.faces_dir):
            person_path = os.path.join(self.faces_dir, person_dir)
            if os.path.isdir(person_path):
                faces = []
                for filename in os.listdir(person_path):
                    if filename.endswith((".jpg", ".png", ".jpeg")):
                        img = cv2.imread(os.path.join(person_path, filename), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            faces.append(img)
                
                if faces:
                    known_faces[person_dir] = faces
        
        print(f"Carregados {len(known_faces)} pessoas com {sum(len(v) for v in known_faces.values())} fotos")
        return known_faces

    def compare_faces(self, face_img, threshold=30):
        """Compara um rosto com os conhecidos usando múltiplas poses"""
        best_match = None
        min_diff = float('inf')
        
        face_img = cv2.resize(face_img, (100, 100))
        face_img = cv2.equalizeHist(face_img)
        
        for name, faces in self.known_faces.items():
            for known_face in faces:
                try:
                    known_face_proc = cv2.resize(known_face, (100, 100))
                    known_face_proc = cv2.equalizeHist(known_face_proc)
                    
                    diff = np.sum((face_img.astype("float") - known_face_proc.astype("float")) ** 2)
                    diff /= float(face_img.shape[0] * face_img.shape[1])
                    
                    if diff < min_diff:
                        min_diff = diff
                        best_match = name
                except:
                    continue
        
        print(f"Diferença mínima: {min_diff:.2f}, Reconhecido: {best_match}")
        return best_match if min_diff < threshold else None

    def save_person(self, name, face_img):
        """Salva uma pessoa com múltiplas poses"""
        person_dir = os.path.join(self.faces_dir, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        # Salva 3 poses diferentes
        timestamp = int(time.time())
        cv2.imwrite(os.path.join(person_dir, f"{name}_frontal_{timestamp}.jpg"), face_img)
        
        # Simula poses olhando para esquerda/direita (na prática, você deve capturar essas imagens)
        h, w = face_img.shape
        M_left = cv2.getRotationMatrix2D((w/2, h/2), 15, 1)  # 15 graus para esquerda
        left_pose = cv2.warpAffine(face_img, M_left, (w, h))
        cv2.imwrite(os.path.join(person_dir, f"{name}_esquerda_{timestamp}.jpg"), left_pose)
        
        M_right = cv2.getRotationMatrix2D((w/2, h/2), -15, 1)  # 15 graus para direita
        right_pose = cv2.warpAffine(face_img, M_right, (w, h))
        cv2.imwrite(os.path.join(person_dir, f"{name}_direita_{timestamp}.jpg"), right_pose)
        
        self.known_faces = self.load_known_faces()  # Recarrega os rostos
        print(f"Pessoa '{name}' salva com 3 poses diferentes")

    def run(self):
        """Executa o sistema de reconhecimento"""
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            print("Não foi possível abrir a câmera!")
            return

        root = Tk()
        root.withdraw()

        frame_count = 0
        print("Instruções:")
        print("- Pressione 's' para salvar um novo rosto (3 poses serão capturadas)")
        print("- Pressione 'q' para sair")

        while True:
            ret, frame = video.read()
            if not ret:
                print("Erro ao capturar frame da câmera")
                break

            frame = cv2.flip(frame, 1)  # Espelha a imagem
            gray = preprocess_image(frame)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            # Atualiza temporizadores
            current_time = time.time()
            for face_id in list(self.face_timers.keys()):
                if current_time - self.face_timers[face_id] > 10:  # 10 segundos de cooldown
                    del self.face_timers[face_id]

            # Processa cada rosto detectado
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                face_id = f"{x}_{y}_{w}_{h}"

                # Verificação periódica
                if frame_count % 30 == 0 or face_id not in self.last_seen:
                    name = self.compare_faces(roi_gray)
                    self.last_seen[face_id] = name
                    
                    if name and (face_id not in self.recognized_names or self.recognized_names[face_id] != name):
                        self.recognized_names[face_id] = name
                        register_attendance(name, self.attendance_file)
                        print(f"Presença registrada: {name}")

                # Exibe o nome reconhecido
                if face_id in self.recognized_names:
                    name = self.recognized_names[face_id]
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
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    roi_gray = gray[y:y+h, x:x+w]
                    
                    name = simpledialog.askstring("Novo Cadastro", "Digite o nome da pessoa:", parent=root)
                    if name:
                        self.save_person(name, roi_gray)
                        messagebox.showinfo("Sucesso", f"{name} cadastrado(a) com sucesso com 3 poses diferentes!", parent=root)

        video.release()
        cv2.destroyAllWindows()

# --- Execução do sistema ---
if __name__ == "__main__":
    recognizer = FaceRecognizer()
    recognizer.run()