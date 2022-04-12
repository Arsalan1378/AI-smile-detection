from mtcnn import MTCNN
from imutils import paths
import cv2

temp_count_round = 0

facedetector = MTCNN()

Raw_images_directories_addresses = list(paths.list_images('/Trainfiles/pictures'))
Raw_images_directories_addresses = sorted(Raw_images_directories_addresses)

for selected_directory_address in Raw_images_directories_addresses:

    temp_image = cv2.imread(selected_directory_address)
    faces = facedetector.detect_faces(temp_image)

    for d in faces:
        x, y, w, h = d['box']

        Cropped_Image = temp_image[y:y + h, x:x + w]
        cv2.imwrite('/Trainfiles/preprocessed_images/' + (selected_directory_address[21:33]), Cropped_Image)

    if temp_count_round % 500 == 0:
        a = '█' * (temp_count_round // 500) * 5
        b = ' ' * (40 - (temp_count_round // 500) * 5)
        print('|' + a + b + '| ', end='\t')
        print(temp_count_round, ' images from 4000 images preprocessed ')

    temp_count_round += 1
