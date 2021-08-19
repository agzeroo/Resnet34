##### 2. Facelandmark, 점 거리 계산 #####
import cv2
import mediapipe as mp
import os
from tqdm import tqdm
from time import sleep
import pandas as pd
import numpy as np
import dlib
import natsort

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

path = 'C:/Users/master15/Desktop/0812resnet/data/'   # 폴더 경로
os.chdir(path)   # 해당 폴더로 이동

# For static images:
IMAGE_FILES = []

files_tmp = os.listdir(path)
# print(files_tmp)
files = natsort.natsorted(files_tmp)
# print(files)

for file in files:
    if '.png' in file:
        f = cv2.imread(file)
        # 이미지 256x256 resize
        f = cv2.resize(f, dsize=(800, 800), interpolation=cv2.INTER_AREA)
        IMAGE_FILES.append(f)
    if '.jpg' in file:
        f = cv2.imread(file)
        f = cv2.resize(f, dsize=(800, 800), interpolation=cv2.INTER_AREA)
        IMAGE_FILES.append(f)

# 코 포인트
nose_points = [169, 7, 198, 196, 6, 5, 2, 20, 95, 123, 197, 4, 237, 199, 132, 50, 103, 130, 65, 99, 241, 100,
               352, 420, 249, 282, 364, 361, 280, 332, 359, 259, 456, 306, 291]

# 코 길이
nose_length = [169, 7, 198, 196, 6, 5, 2, 20, 95]
# 콧대 폭
R_width = [352, 420, 457, 421]
L_width = [123, 197, 237, 199]

# 콧볼
R_nostril = [361, 280, 279, 295, 456, 461]

L_nostril = [132, 50, 103, 65, 99, 241]

# 코
# 코 길이
nose_length_dict = {}

# 콧대 폭
# 오른쪽
R_width_dict = {}
# 왼쪽
L_width_dict = {}

# 콧볼
# 오른쪽
R_nostril_dict = {}

# 왼쪽
L_nostril_dict = {}


# 코 함수
length_list = []
width_list = []
nostril_list = []


def nose(image):

    # ------------------------
    # 코 길이
    length = nose_length_dict['95'][1]-nose_length_dict['169'][1]
    print(length)
    length_list.append(length)

    # ------------------------
    # 코 폭
    width = R_width_dict['421'][0] - \
        L_width_dict['199'][0]
    print(width)
    width_list.append(width)

    # ------------------------
    # 콧볼 폭

    nostril = R_nostril_dict['279'][0] - \
        L_nostril_dict['103'][0]
    print(nostril)
    nostril_list.append(nostril)


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.8) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
        image = file
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            # print('face_landmarks:', face_landmarks)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
        cv2.imwrite('./tmp/annotated_image_' +
                    str(idx) + '.jpg', annotated_image)
        if not cv2.imwrite('./tmp/annotated_image_' + str(idx) + '.jpg', annotated_image):
            raise Exception("Could not write image")

        num = 1
        for face in results.multi_face_landmarks:
            for (i, landmark) in enumerate(face.landmark):
                x = landmark.x
                y = landmark.y
                z = landmark.z
                #print(f'{num}번째 랜드마크', x, y, z)

                shape = image.shape
                relative_x = int(x * shape[1])
                relative_y = int(y * shape[0])

                # print(relative_x, relative_y)

                cv2.circle(image, (relative_x, relative_y),
                           radius=1, color=(0, 255, 0), thickness=1)

                # -------------------------------
                # 코
                if num in nose_length:
                    nose_length_dict[f'{num}'] = relative_x, relative_y

                elif num in R_width:
                    R_width_dict[f'{num}'] = relative_x, relative_y

                elif num in L_width:
                    L_width_dict[f'{num}'] = relative_x, relative_y

                elif num in R_nostril:
                    R_nostril_dict[f'{num}'] = relative_x, relative_y
                elif num in L_nostril:
                    L_nostril_dict[f'{num}'] = relative_x, relative_y

                # -------------------------------

                num += 1

                cv2.putText(image, "{}".format(i + 1), (relative_x, relative_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            # 넘버링 이미지 저장 코드(1~2분 걸림)
            cv2.imwrite('./numbering/numbering_' + str(idx) + '.jpg', image)

        #cv2.imshow('dd', image)
        # cv2.waitKey(0)

        print('='*60)
        nose(image)

        nose_dict = {"코 길이": length_list,
                     "코 폭": width_list, "콧볼 폭": nostril_list}

        nose_pd = pd.DataFrame(nose_dict)


print(nose_pd)

# csv파일로 저장
nose_pd.to_csv("nose_0812_edit.csv", encoding='utf-8-sig')
