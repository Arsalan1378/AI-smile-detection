from imutils import paths
import cv2


facedetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Raw_images_directories_addresses = list(paths.list_images('AI-smile-detection/Trainfiles/pictures'))
Raw_images_directories_addresses = sorted(Raw_images_directories_addresses)
counted_rounds = 0

for selected_directory_address in Raw_images_directories_addresses:
    counted_rounds += 1

    temp_image = cv2.imread(selected_directory_address)
    faces = facedetector.detectMultiScale(cv2.medianBlur(cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY), 3), 1.3, 5)

    for (x, y, w, h) in faces:

        Cropped_Image = temp_image[y:y + h, x:x + w]
        cv2.imwrite('AI-smile-detection/Trainfiles/preprocessed_images_CascadeClassifier_Medianblur/' + (selected_directory_address[-12:]),
                    Cropped_Image)
        print(selected_directory_address, end=',')

    if counted_rounds % 500 == 0:
        a = '█' * (counted_rounds // 500) * 5
        b = ' ' * (40 - (counted_rounds // 500) * 5)
        print()
        print('|' + a + b + '| ', end='\t')
        print(counted_rounds, ' images from 4000 images preprocessed ')