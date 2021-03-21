import cv2
import numpy as np

img = cv2.imread('images/girl.jpg')


def resizing(new_width=None, new_height=None, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]

    if new_width is None and new_height is None:
        return img

    if new_width is None:
        ratio = new_height / h
        dimension = (int(w * ratio), new_height)

    else:
        ratio = new_width / w
        dimension = (new_width, int(h * ratio))

    res_img = cv2.resize(img, dimension, interpolation=interp)
    cv2.imwrite('new_girl_resize.jpg', res_img)

    # res_img = cv2.resize(img, (500, 900), cv2.INTER_NEAREST)
    # cv2.imwrite('res_girl.jpg', res_img)
    res_img_nearest = cv2.resize(img, (int(w / 1.4), int(h / 1.4)), cv2.INTER_NEAREST)
    res_img_linear = cv2.resize(img, (int(w / 1.4), int(h / 1.4)), cv2.INTER_LINEAR)
    # cv2.imwrite('girl_nearest.jpg', res_img_nearest)
    # cv2.imwrite('girl_linear.jpg', res_img_linear)


def shifting():
    h, w = img.shape[:2]
    translation_matrix = np.float32([[1, 0, 200], [0, 1, 300]])
    dst = cv2.warpAffine(img, translation_matrix, (w, h))
    cv2.imwrite('girl_right_and_down_1.jpg', dst)


def cropping():
    crop_img = img[10:450, 300:750]
    cv2.imwrite('crop_face.png', crop_img)


def rotation():
    (h, w) = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 0.6)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    cv2.imwrite('rotated_girl.jpg', rotated)
