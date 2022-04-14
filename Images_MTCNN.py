from imutils import paths
import cv2
from mtcnn import MTCNN

facedetector = MTCNN()

Raw_images_directories_addresses = list(paths.list_images('/Trainfiles/pictures'))
Raw_images_directories_addresses = sorted(Raw_images_directories_addresses)
counted_rounds = 0

for selected_directory_address in Raw_images_directories_addresses:
    counted_rounds += 1

    temp_image = cv2.imread(selected_directory_address)
    faces = facedetector.detect_faces(temp_image)

    for face_box_Coordinates in faces:
        x, y, w, h = face_box_Coordinates['box']

        Cropped_Image = temp_image[y:y + h, x:x + w]
        cv2.imwrite('/Trainfiles/preprocessed_images/' + (selected_directory_address[-12:]), Cropped_Image)
        print(selected_directory_address, end=',')

    if counted_rounds % 500 == 0:
        a = '█' * (counted_rounds // 500) * 5
        b = ' ' * (40 - (counted_rounds // 500) * 5)
        print()
        print('|' + a + b + '| ', end='\t')
        print(counted_rounds, ' images from 4000 images preprocessed ')


# Phase two for recognized faces

for scale_factor_float_number in [0.80, 0.95, 0.99]:

    Preprossesd_images_directories_addresses = list(paths.list_images('/Trainfiles/preprocessed_images/'))

    Raw_images_name_set = {directories_addresses[-12:] for directories_addresses in Raw_images_directories_addresses}
    Preprossesd_images_name_set = {directories_addresses[-12:] for directories_addresses in
                                   Preprossesd_images_directories_addresses}

    Unprossed_images_by_MTCNN = sorted(list(Raw_images_name_set.difference(Preprossesd_images_name_set)))
    Unprossed_images_by_MTCNN_directories_addresses = [(Raw_images_directories_addresses[0][:-12]) + images_name for
                                                       images_name in Unprossed_images_by_MTCNN]

    counted_rounds = 0

    facedetector_Phase_two = MTCNN(scale_factor=scale_factor_float_number)
    for selected_directory_address in Unprossed_images_by_MTCNN_directories_addresses:
        counted_rounds += 1

        temp_image = cv2.imread(selected_directory_address)
        faces = facedetector_Phase_two.detect_faces(temp_image)

        for face_box_Coordinates in faces:
            x, y, w, h = face_box_Coordinates['box']

            cv2.imwrite('/Trainfiles/preprocessed_images/' + (selected_directory_address[-12:]), Cropped_Image)
            print(selected_directory_address, end=',')

    print()
    print(counted_rounds, f'images from {len(Unprossed_images_by_MTCNN_directories_addresses)} images preprocessed')




