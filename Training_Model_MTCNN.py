from imutils import paths
import cv2
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from skimage.feature import multiblock_lbp as Multi_LBP
import joblib
import warnings

warnings.filterwarnings('ignore')

Raw_images_directories_addresses = list(paths.list_images('Trainfiles/preprocessed_images_MTCNN'))
Raw_images_directories_addresses = sorted(Raw_images_directories_addresses)
temp_number = 0
temp_count_round = 1
run_time_list = []
for selected_directory_address in Raw_images_directories_addresses[:10]:

    start_time = time.time()

    temp_images_parameters = []
    temp_image_LBP_list = []

    temp_image = cv2.imread(selected_directory_address, 0)
    temp_image = cv2.medianBlur(temp_image, 3)
    temp_image = cv2.resize(temp_image, (64, 64))

    temp_image_LBP_list = []
    for rows in list(range(0, 65, 3)):
        for column in list(range(0, 65, 3)):
            temp_image_LBP_multi_block = Multi_LBP(temp_image, rows, column, 4, 4)
            temp_image_LBP_list.append(temp_image_LBP_multi_block)

    temp_images_parameters.extend(temp_image_LBP_list)

    if temp_number == 0:
        images_parameters = temp_images_parameters
        temp_number = 1

    elif temp_number:
        images_parameters = np.vstack((images_parameters, temp_images_parameters))

    if temp_count_round % 200 == 0:
        a = '█' * (temp_count_round // 200) * 2
        b = ' ' * (40 - (temp_count_round // 200) * 2)

        print('|' + a + b + '| ', end='\t')
        print(temp_count_round, ' images from 4000 images preprocessed ')

    temp_count_round += 1
    end_time = time.time()
    run_time_list.append(end_time - start_time)

print(f'Average run time for every image : {np.mean(run_time_list)} sec')
print(f'Total run time : {sum(run_time_list)}')

Raw_images_directories_addresses = list(paths.list_images('Trainfiles/pictures'))
Raw_images_directories_addresses = sorted(Raw_images_directories_addresses)

Preprossesd_images_directories_addresses = list(paths.list_images('Trainfiles/preprocessed_images_MTCNN'))

Raw_images_name_set = {directories_addresses[-12:] for directories_addresses in Raw_images_directories_addresses}
Preprossesd_images_name_set = {directories_addresses[-12:] for directories_addresses in
                               Preprossesd_images_directories_addresses}

Unprossed_images_by_MTCNN = sorted(list(Raw_images_name_set.difference(Preprossesd_images_name_set)))

images_index_number = {int(image_index_number[-8:-4]) - 1 for image_index_number in Unprossed_images_by_MTCNN}

images_parameters_labels_temp = pd.read_csv('Trainfiles/labels.csv')
images_parameters_labels_temp = images_parameters_labels_temp['class']

images_parameters_labels = pd.DataFrame(columns=["class"])
counter = 0
for index_number, value in enumerate(images_parameters_labels_temp):
    index_number_set = {index_number}

    if index_number_set.issubset(images_index_number):
        counter += 1
    else:
        Dict_temp = {"class": value}
        images_parameters_labels = images_parameters_labels.append(Dict_temp, ignore_index=True)

images_parameters_labels = list(images_parameters_labels["class"])
print(f'deleted rows = {counter}')
print(images_parameters_labels)

acc_tst = 0
rounds = 0
while True:
    # start_time = time.time()
    X_train, X_test, Y_train, Y_test = train_test_split(images_parameters, images_parameters_labels)

    model = svm.SVC(kernel='rbf', C=1, cache_size=10000)

    model.fit(X_train, Y_train)

    ACC_test = model.score(X_test, Y_test)
    rounds += 1
    # end_time = time.time()
    # print(end_time - start_time)

    if acc_tst <= ACC_test:
        acc_tst = ACC_test
        print(acc_tst)

    if rounds % 100 == 0:
        print(rounds)

    if ACC_test >= 0.89:
        break

filename = 'filename.sav'
joblib.dump(model, filename)

