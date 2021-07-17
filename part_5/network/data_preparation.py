import cv2
import shutil
import os


# 1. Одинаковое количество файлов во всех папках
# 2. Разделить выборку на тестовые и тренировочные: 20 на 80
# 3. Привести все фото к ч/б
# 4. Привести всё к одному размеру - 28 на 28


def remove_files():
    path_to_dataset = 'dataset/train/'
    for folder in os.listdir(path_to_dataset):
        count = 0
        needed_count_of_files = 34571
        path_to_folder = path_to_dataset+folder+'/'
        images = os.listdir(path_to_folder)
        count_for_remove = len(images) - needed_count_of_files
        for image in images:
            if count == count_for_remove:
                break
            else:
                os.remove(path_to_folder+image)
                count += 1


def split_dataset():
    path_to_train_dataset = 'dataset/train/'
    path_to_test_dataset = 'dataset/test/'
    for folder in os.listdir(path_to_train_dataset):
        count = 0
        count_for_move = 6500
        path_to_train_folder = path_to_train_dataset+folder+'/'
        path_to_test_folder = path_to_test_dataset+folder+'/'
        for image in os.listdir(path_to_train_folder):
            if count == count_for_move:
                break
            else:
                shutil.move(path_to_train_folder+image, path_to_test_folder+image)
                count+=1


def image_binarization():
    #'dataset/train/'
    path_to_dataset = 'dataset/test/'
    for folder in os.listdir(path_to_dataset):
        path_to_folder = path_to_dataset+folder+'/'
        images = os.listdir(path_to_folder)
        for image in images:
            img = cv2.imread(path_to_folder+image, cv2.IMREAD_GRAYSCALE)
            (thresh, im_bn) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cv2.imwrite(path_to_folder+image, im_bn)


def resize_images():
    #'dataset/test/'
    path_to_dataset = 'dataset/train/'
    for folder in os.listdir(path_to_dataset):
        path_to_folder = path_to_dataset + folder + '/'
        images = os.listdir(path_to_folder)
        for image in images:
            img = cv2.imread(path_to_folder+image)
            res_img = cv2.resize(img, (28, 28))
            cv2.imwrite(path_to_folder+image, res_img)



