from tensorflow import keras
import numpy as np
import itertools
import functools
import cv2

# Проблемы:
# 1.Сортировка контуров
# 2. Контур в контуре при распозновании чисел
# 3. Убрали линии колонки, так как не находились все цифры - find_frame
# 4. Выравнивание таблицы


document_columns_dict = {
    "Name_0": False, "Account_0": False, "Account_1": True,
    "Adress_0": False, "Adress_1": False, "Number_GVS_0": False,
    "Number_GVS_1": True, "Number_GVS_2": True, "Number_GVS_3": True,
    "GVS_0": False, "GVS_1": True, "GVS_2": True, "GVS_3": True,
    "NUMBER_XVS_0": False, "NUMBER_XVS_1": True, "NUMBER_XVS_2": True,
    "NUMBER_XVS_3": True, "XVS_0": False, "XVS_1": True, "XVS_2": True,
    "XVS_3": True, "Date&Phone_0": False, "Date&Phone_1": False,
    "Date&Phone_2": False, "Date&Phone_3": False
}


# https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
def skew_correction(image_name):
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    _, img_bin = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(img_bin > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)

    elif angle == 90:
        angle = 0

    elif 45 < angle < 90:
        angle = 90 - angle

    else:
        angle = -angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# https://stackoverflow.com/questions/41667002/sorting-tuples-with-a-custom-sorting-function-using-a-range
def custom_tuple_sorting(s, t, offset=4):
    x0, y0, _, _ = s
    x1, y1, _, _ = t
    if abs(y0 - y1) > offset:
        if y0 < y1:
            return -1
        else:
            return 1
    else:
        if x0 < x1:
            return -1

        elif x0 == x1:
            return 0

        else:
            return 1


def sort_contours(cnts, method):
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]

    if method == "left-to-right":
        bounding_boxes.sort(key=lambda tup: tup[0])

    elif method == "top-to-right":
        bounding_boxes.sort(key=functools.cmp_to_key(lambda s, t: custom_tuple_sorting(s, t)))
    return bounding_boxes


def find_main_lines(image, type):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    if type == 'h':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    elif type == 'v':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    erode_image = cv2.erode(img_bin, kernel, iterations=1)
    dilate_image = cv2.dilate(erode_image, kernel, iterations=1)
    return dilate_image


def merge_lines(horizontal_lines, vertical_lines):
    kernel = np.ones((3, 3), np.uint8)
    merge_image = horizontal_lines + vertical_lines
    merge_image = cv2.dilate(merge_image, kernel, iterations=2)
    return merge_image


def find_contours_of_table(image):
    _, cnts, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def find_frame(frame_image, crop_image):
    black_pixels = np.where(frame_image == 255)
    y = black_pixels[0]
    x = black_pixels[1]
    for i in range(len(y)):
        crop_image[y[i]][x[i]] = 255
    return crop_image


def image_binarization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    return img_bin


# контур в контуре
def value_compare(x00, y00, x01, y01, x10, y10, x11, y11):
    if x10 >= x00 and y10 >= y00 and x11 <= x01 and y11 <= y01:
        in_rec = [x10, y10, x11, y11]
        return in_rec


def find_digit_coordinates(image):
    _, cnts, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = sort_contours(cnts, method="left-to-right")[1:]
    all_contours = []
    for i in range(0, len(bounding_boxes)):
        x, y, w, h = bounding_boxes[i][0], bounding_boxes[i][1], bounding_boxes[i][2], bounding_boxes[i][3]
        if h > 20 and w > 10:
            digit_coordinates = [x, y, x + w, y + h]
            all_contours.append(digit_coordinates)
    return all_contours


def predicting(image):
    img = keras.preprocessing.image
    model = keras.models.load_model('network/models/model.h5')
    x = img.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    return np.argmax(classes[0])


def crop_digit(image, x0, y0, x1, y1):
    img_crop = image[y0 - 2:y1 + 2, x0 - 2:x1 + 2]
    res_crop_img = cv2.resize(img_crop, (28, 28))
    prediction_digit = predicting(res_crop_img)
    return prediction_digit


def detect_contour_in_contours(all_contours):
    for rec1, rec2 in itertools.permutations(all_contours, 2):
        in_rec = value_compare(rec1[0], rec1[1], rec1[2], rec1[3],
                               rec2[0], rec2[1], rec2[2], rec2[3])
        if in_rec is not None:
            all_contours.remove(in_rec)
    return all_contours


def find_coordinates_of_rows(rotated_image, frame_image, cnts):
    bounding_boxes = sort_contours(cnts, "top-to-right")
    count = 0
    for i in range(0, len(bounding_boxes)):
        x, y, w, h = bounding_boxes[i][0], bounding_boxes[i][1], bounding_boxes[i][2], bounding_boxes[i][3]
        if (w < bounding_boxes[0][2] and h > bounding_boxes[0][3] / 11) and w > 3 * h:
            if list(document_columns_dict.values())[count]:

                img_crop = rotated_image[y - 12:y + h + 12,
                                         x - 6:x + w + 6]

                frame_crop = frame_image[y - 12:y + h + 12,
                                         x - 6:x + w + 6]

                img_crop = find_frame(frame_crop, img_crop)
                image_bin = image_binarization(img_crop)
                contours_arr = find_digit_coordinates(image_bin)
                right_digits = detect_contour_in_contours(contours_arr)
                s = ""
                for rec in right_digits:
                    prediction = crop_digit(image_bin, rec[0], rec[1], rec[2], rec[3])
                    s += str(prediction)
                document_columns_dict.update({list(document_columns_dict.keys())[count]: s})
            else:
                document_columns_dict.update({list(document_columns_dict.keys())[count]: ''})
            count += 1
    print(document_columns_dict)


if __name__ == '__main__':
    correct_image = skew_correction('table_images/4.jpg')
    detected_horry_lines = find_main_lines(correct_image, 'h')
    detected_verti_lines = find_main_lines(correct_image, 'v')
    united_image = merge_lines(detected_verti_lines, detected_horry_lines)
    contours = find_contours_of_table(united_image)
    find_coordinates_of_rows(correct_image, united_image, contours)