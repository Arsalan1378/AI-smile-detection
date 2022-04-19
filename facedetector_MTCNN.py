from skimage.feature import multiblock_lbp as Multi_LBP
import cv2
from mtcnn import MTCNN
import numpy as np
import time
from joblib import load
import warnings

warnings.filterwarnings('ignore')


def features_extraction(image):

    image_LBP_list = []

    image = image[:, :, 0]
    image = cv2.medianBlur(image, 3)
    image = cv2.resize(image, (64, 64))

    for rows in list(range(0, 65, 3)):
        for column in list(range(0, 65, 3)):
            image_LBP_multi_block = Multi_LBP(image, rows, column, 4, 4)
            image_LBP_list.append(image_LBP_multi_block)

    return np.array(image_LBP_list).reshape(1, -1)


model_directory = 'svm_rbf_c1_version_5_just_MBLBP_(modify_dataset).sav'
model = load(model_directory)
facedetector = MTCNN()
cap = cv2.VideoCapture("http://192.168.137.214:8080/video")
counter = 0

while True:
    start_time = time.time()

    ret, frame = cap.read()

    frame = cv2.resize(frame, (720, 480))
    counter += 1
    if counter == 1:
        frame_1 = frame
        frame_lable = [0]

    if ret and (counter % 10 == 0):

        faces = facedetector.detect_faces(frame)
        for face_box_Coordinates in faces:
            x, y, w, h = face_box_Coordinates['box']
            Cropped_frame = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 30, 250), 2)
            start_time_1 = time.time()
            extracted_features = features_extraction(Cropped_frame)
            frame_lable = model.predict(extracted_features)
            print(frame_lable)

            end_time_1 = time.time()
            print(f'Model run time : {end_time_1 - start_time_1}')

            if int(frame_lable[0]) == 1:
                frame_with_lable = cv2.putText(frame, 'S', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                               (0, 70, 150), 2, cv2.LINE_AA)
            else:
                frame_with_lable = cv2.putText(frame, 'N', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                               (0, 70, 150), 2, cv2.LINE_AA)
            frame_1 = frame_with_lable

        cv2.imshow('', frame_with_lable)
        cv2.waitKey(1)

        end_time = time.time()
        print(f'Total run time : {end_time - start_time}')

    elif not ret:
        break

    else:
        if counter >= 10:
            RN = np.random.randint(low=-2, high=2)
            x, y, w, h = x+RN, y+RN, w+RN, h+RN
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 30, 250), 2)
        if int(frame_lable[0]) == 1:
            frame_1 = cv2.putText(frame, 'S', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                  (0, 70, 150), 2, cv2.LINE_AA)
        else:
            frame_1 = cv2.putText(frame, 'N', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                  (0, 70, 150), 2, cv2.LINE_AA)
        cv2.imshow('', frame_1)
        cv2.waitKey(1)

cap.release()