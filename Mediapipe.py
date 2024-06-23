import mediapipe as mp
import cv2 as cv
from scipy.spatial import distance as dis
import threading
import pyttsx3
import datetime
import base64
import requests
import json
import time

COLOR_WHITE = (255,255,255)
ID_DEVICE = '001'

frame_count = 0
min_frame = 6
min_tolerance = 5.0
last_capture_time = 0
capture_interval = 10  # Intervalo de 10 segundos

face_mesh = mp.solutions.face_mesh
draw_utils = mp.solutions.drawing_utils
landmark_style = draw_utils.DrawingSpec((0,255,0), thickness=1, circle_radius=1)
connection_style = draw_utils.DrawingSpec((0,0,255), thickness=1, circle_radius=1)

# Pontos de referência para os lábios
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

# Pontos de referência para o olho direito
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]

# Pontos de referência para o olho esquerdo
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]

# Pontos de referência para a linha superior e inferior do olho esquerdo
LEFT_EYE_TOP_BOTTOM = [386, 374]
LEFT_EYE_LEFT_RIGHT = [263, 362]

# Pontos de referência para a linha superior e inferior do olho direito
RIGHT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_LEFT_RIGHT = [133, 33]

# Pontos de referência para a linha superior e inferior dos lábios
UPPER_LOWER_LIPS = [13, 14]
LEFT_RIGHT_LIPS = [78, 308]

# Pontos de referência para o contorno do rosto
FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]


def run_speech(speech, speech_message):
    # Função para reproduzir a mensagem de aviso
    speech.say(speech_message)
    speech.runAndWait()
    

def draw_landmarks(image, outputs, land_mark, color):
    height, width = image.shape[:2]
             
    for face in land_mark:
        # Itera sobre os pontos de referência do rosto
        point = outputs.multi_face_landmarks[0].landmark[face]
        
        # Escala as coordenadas normalizadas para as dimensões da imagem
        point_scale = ((int)(point.x * width), (int)(point.y*height))
        
        # Desenha um círculo no ponto de referência
        cv.circle(image, point_scale, 2, color, 1)


def euclidean_distance(image, top, bottom):
    height, width = image.shape[0:2]
            
    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)
    
    # Calcula a distância euclidiana entre dois pontos
    distance = dis.euclidean(point1, point2)
    return distance


def get_aspect_ratio(image, outputs, top_bottom, left_right):
    landmark = outputs.multi_face_landmarks[0]
            
    top = landmark.landmark[top_bottom[0]]
    bottom = landmark.landmark[top_bottom[1]]
    
    # Calcula a distância entre os pontos superior e inferior
    top_bottom_dis = euclidean_distance(image, top, bottom)
    
    left = landmark.landmark[left_right[0]]
    right = landmark.landmark[left_right[1]]
    
    # Calcula a distância entre os pontos esquerdo e direito
    left_right_dis = euclidean_distance(image, left, right)
    
    # Calcula a relação de aspecto entre as distâncias esquerda-direita e superior-inferior
    aspect_ratio = left_right_dis/ top_bottom_dis
    
    return aspect_ratio

def post_image(id_device, image_path):
    with open(image_path, "rb") as image_file:
        # Converte a imagem para base64
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    data = {
        "idDevice": id_device,
        "image": encoded_string,
        "type": "SLEEPING",
        "occurrenceDate": datetime.datetime.now().isoformat()
    }

    url = "https://drivewatchbackend-production.up.railway.app/api/v1/register"
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")


# Inicializa o modelo de detecção de rosto
face_model = face_mesh.FaceMesh(static_image_mode = False,
                                max_num_faces = 2,
                                min_detection_confidence = 0.6,
                                min_tracking_confidence = 0.5)

# Inicializa a captura de vídeo
capture = cv.VideoCapture(0)

# Inicializa o mecanismo de fala
speech = pyttsx3.init()

while True:
    # Captura um frame do vídeo
    result, image = capture.read()
    
    if result:
        # Cria uma cópia da imagem
        clean_image = image.copy()

        # Converte o frame para o espaço de cores RGB
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Processa o frame para detectar rostos
        outputs = face_model.process(image_rgb)

        if outputs.multi_face_landmarks:     
            
            # Desenha os pontos de referência do olho esquerdo
            draw_landmarks(image, outputs, LEFT_EYE_TOP_BOTTOM, COLOR_WHITE)
            draw_landmarks(image, outputs, LEFT_EYE_LEFT_RIGHT, COLOR_WHITE)
            ratio_left =  get_aspect_ratio(image, outputs, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
            
            # Desenha os pontos de referência do olho direito
            draw_landmarks(image, outputs, RIGHT_EYE_TOP_BOTTOM, COLOR_WHITE)
            draw_landmarks(image, outputs, RIGHT_EYE_LEFT_RIGHT, COLOR_WHITE)
            ratio_right =  get_aspect_ratio(image, outputs, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
            
            # Calcula a média da relação de aspecto entre os olhos esquerdo e direito
            ratio = (ratio_left + ratio_right) / 2.0
            
            if ratio > min_tolerance:
                frame_count += 1
            else:
                frame_count = 0
                
            current_time = time.time()
            if frame_count > min_frame and (current_time - last_capture_time) > capture_interval:
                # Olhos fechados
                message = 'You are sleeping, stop the vehicle'
                # Salva a imagem capturada
                image_path = f'{ID_DEVICE}_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.jpg'
                cv.imwrite(image_path, clean_image)
                
                # Atualiza o tempo da última captura
                last_capture_time = current_time

                # Cria uma nova thread para enviar a imagem
                threading.Thread(target=post_image, args=(ID_DEVICE, image_path)).start()
                # Cria uma nova thread para reproduzir a mensagem de aviso
                threading.Thread(target=run_speech, args=(speech, message)).start()

            # Desenha os pontos de referência dos labios
            draw_landmarks(image, outputs, UPPER_LOWER_LIPS , COLOR_WHITE)
            draw_landmarks(image, outputs, LEFT_RIGHT_LIPS, COLOR_WHITE)
            
            # Calcula se a boca está aberta
            ratio_lips =  get_aspect_ratio(image, outputs, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)

            if ratio_lips < 1.8:
                # Boca aberta
                message = 'You look tired, take a break to rest'
                threading.Thread(target=run_speech, args=(speech, message)).start()
           
        cv.imshow("Drive Watch", image)
        if cv.waitKey(1) & 255 == 27:
            break
        
capture.release()
cv.destroyAllWindows()
