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
import numpy as np

ID_DEVICE = '001'

COLOR_WHITE = (255,255,255)
COLOR_RED = (0, 0, 255)  
COLOR_GREEN = (0, 255, 0)  

min_frameEyes = 11
min_frameMouth = 35
min_toleranceEyes = 5.3
min_toleranceMouth = 1.4
last_capture_time = 0
capture_interval = 10  # Intervalo de 10 segundos entre os envios para o servidor

face_mesh = mp.solutions.face_mesh
draw_utils = mp.solutions.drawing_utils
landmark_style = draw_utils.DrawingSpec((0,255,0), thickness=1, circle_radius=1)
connection_style = draw_utils.DrawingSpec((0,0,255), thickness=1, circle_radius=1)

# Pontos de referência para a linha superior e inferior do olho esquerdo
LEFT_EYE_TOP_BOTTOM = [386, 374]
LEFT_EYE_LEFT_RIGHT = [263, 362]

# Pontos de referência para a linha superior e inferior do olho direito
RIGHT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_LEFT_RIGHT = [133, 33]

# Pontos de referência para a linha superior e inferior dos lábios
UPPER_LOWER_LIPS = [13, 14]
LEFT_RIGHT_LIPS = [78, 308]


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
    image = cv.imread(image_path)
    max_width = 300
    max_height = 300
    if image is None:
        print(f"Failed to load image: {image_path}")
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_image = cv.resize(image, new_size, interpolation=cv.INTER_AREA)
        cv.imwrite(image_path, resized_image)
        
    
    with open(image_path, "rb") as image_file:
        # Converte a imagem para base64
        #encoded_string = base64.b64encode(image_file.read())
        #encoded_string = encoded_string.decode('utf-8')
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        #encoded_string = str(base64.b64encode(open(image_path,"rb").read()).decode("ascii"))
        

        print(encoded_string)
        
        

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
                                max_num_faces = 1,
                                min_detection_confidence = 0.5,
                                min_tracking_confidence = 0.5)

# Inicializa a captura de vídeo
capture = cv.VideoCapture(0)

# Inicializa o mecanismo de fala
speech = pyttsx3.init()

while capture.isOpened():
    # Captura um frame do vídeo
    result, image = capture.read()
    
    if result:
        # Converte o frame para o espaço de cores RGB
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        #image_rgb = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)

        # Cria uma cópia da imagem
        clean_image = image.copy()

        frame_h, frame_w, _ = image.shape
        
        image_rgb.flags.writeable = False

        outputs = face_model.process(image_rgb)

        image_rgb.flags.writeable = True

        image_rgb = cv.cvtColor(image_rgb,cv.COLOR_RGB2BGR)

        img_h , img_w, img_c = image_rgb.shape
        face_2d = []
        face_3d = []

        DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
        
        # Processa o frame para detectar rostos
        outputs = face_model.process(image_rgb)

        if outputs.multi_face_landmarks:   

            # -------------------------------Inicio blobo direção cabeça-------------------------------
            for face_landmarks in outputs.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                        if idx ==1:
                            nose_2d = (lm.x * img_w,lm.y * img_h)
                            nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
                        x,y = int(lm.x * img_w),int(lm.y * img_h)

                        face_2d.append([x,y])
                        face_3d.append(([x,y,lm.z]))


                #Get 2d Coord
                face_2d = np.array(face_2d,dtype=np.float64)

                face_3d = np.array(face_3d,dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length,0,img_h/2],
                                    [0,focal_length,img_w/2],
                                    [0,0,1]])
                distortion_matrix = np.zeros((4,1),dtype=np.float64)

                result,rotation_vec,translation_vec = cv.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)


                #getting rotational of face
                rmat,jac = cv.Rodrigues(rotation_vec)

                angles,mtxR,mtxQ,Qx,Qy,Qz = cv.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                #here based on axis rot angle is calculated 
                text=""

                if x < -18:
                    text="Pare o veiculo"
                elif x < -11.5:
                    text="Sinal de sonolencia"
                
                

                nose_3d_projection,jacobian = cv.projectPoints(nose_3d,rotation_vec,translation_vec,cam_matrix,distortion_matrix)

                p1 = (int(nose_2d[0]),int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] -x *10))

                cv.line(image,p1,p2,(255,0,0),3)

                cv.putText(image,text,(20,50),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
                cv.putText(image,"x: " + str(np.round(x,2)),(500,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                cv.putText(image,"y: "+ str(np.round(y,2)),(500,100),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                cv.putText(image,"z: "+ str(np.round(z, 2)), (500, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # -------------------------------Fim blobo direção cabeça-------------------------------
            
            # Desenha os pontos de referência do olho esquerdo
            draw_landmarks(image, outputs, LEFT_EYE_TOP_BOTTOM, COLOR_GREEN)
            draw_landmarks(image, outputs, LEFT_EYE_LEFT_RIGHT, COLOR_WHITE)
            ratio_left =  get_aspect_ratio(image, outputs, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
            
            # Desenha os pontos de referência do olho direito
            draw_landmarks(image, outputs, RIGHT_EYE_TOP_BOTTOM, COLOR_GREEN)
            draw_landmarks(image, outputs, RIGHT_EYE_LEFT_RIGHT, COLOR_WHITE)
            ratio_right =  get_aspect_ratio(image, outputs, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)

            # Desenha os pontos de referência dos labios
            draw_landmarks(image, outputs, UPPER_LOWER_LIPS , COLOR_GREEN)
            draw_landmarks(image, outputs, LEFT_RIGHT_LIPS, COLOR_WHITE)
            
            # Calcula a média da relação de aspecto entre os olhos esquerdo e direito
            ratio = (ratio_left + ratio_right) / 2.0
            cv.putText(image, "EAR: {:.2f}".format(ratio), (480, 30),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if ratio > min_toleranceEyes:
                frame_count += 1
                if frame_count > 2:
                    cv.putText(image, "Eyes Closed ", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                frame_count = 0
                
            current_time = time.time()
            if frame_count > min_frameEyes and (current_time - last_capture_time) > capture_interval and flagBocejo == 0:
                # Olhos fechados
                message = 'You are sleeping, stop the vehicle'
                cv.putText(image, "DROWSINESS ALERT!", (10, 50),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Salva a imagem capturada
                image_path = f'{ID_DEVICE}_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.jpg'
                cv.imwrite(image_path, clean_image)
                
                # Atualiza o tempo da última captura
                last_capture_time = current_time

                # Cria uma nova thread para enviar a imagem
                threading.Thread(target=post_image, args=(ID_DEVICE, image_path)).start()
                # Cria uma nova thread para reproduzir a mensagem de aviso
                threading.Thread(target=run_speech, args=(speech, message)).start()

            
            # Calcula se a boca está aberta
            ratio_lips =  get_aspect_ratio(image, outputs, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
            cv.putText(image, "MAR: {:.2f}".format(ratio_lips), (480, 60),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if ratio_lips < min_toleranceMouth:
                frame_countM += 1
                # Boca aberta
                cv.putText(image, "Boca aberta", (10, 50),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                flagBocejo = 1
            else:
                frame_countM = 0
                flagBocejo = 0

            if frame_countM > min_frameMouth:
                # Boca aberta
                message = 'You look tired, take a break'
                threading.Thread(target = run_speech, args = (speech, message)).start()
                cv.putText(image, "Bocejo", (10, 80),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            '''
            draw_utils.draw_landmarks(image=image_rgb,
                                    landmark_list=face_landmarks,
                                    connections=face_mesh.FACEMESH_CONTOURS,
                                    landmark_drawing_spec=landmark_style,
                                    connection_drawing_spec=landmark_style)
            '''
           
        cv.imshow("Drive Watch", image)
        
        if cv.waitKey(1) & 255 == 27:
            break
        
capture.release()
cv.destroyAllWindows()