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


model_directory = 'svm_rbf_c1_version_6_just_MBLBP_(modify_dataset)_haar.sav'
model = load(model_directory)
facedetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture("http://192.168.137.99:8080/video")
counter = 0
true_lable = 0

while True:
    start_time = time.time()

    ret, frame = cap.read()

    frame = cv2.resize(frame, (720, 360))
    if counter == 1:
        frame_1 = frame
        frame_lable = [0]

    if ret:

        faces = facedetector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5)
        for (x, y, w, h) in faces:

            Cropped_frame = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 30, 250), 2)
            start_time_1 = time.time()
            extracted_features = features_extraction(Cropped_frame)
            frame_lable = model.predict(extracted_features)
            print(frame_lable)

            end_time_1 = time.time()
            print(f'Model run time : {end_time_1 - start_time_1}')
            if frame_lable[0] == 0:
                true_lable += 1
            else:
                true_lable = 0

            if int(frame_lable[0]) == 1 :
                frame_with_lable = cv2.putText(frame, 'S', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                               (0, 70, 150), 2, cv2.LINE_AA)
            else:
                frame_with_lable = cv2.putText(frame, 'N', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                               (0, 70, 150), 2, cv2.LINE_AA)

        cv2.imshow('', frame_with_lable)
        cv2.waitKey(1)

        end_time = time.time()
        print(f'Total run time : {end_time - start_time}')

    elif not ret:
        break

cap.release()
