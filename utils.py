import time

import cv2


def draw_bbox(image, position):
    x, y, w, h = position['x'], position['y'], position['w'], position['h']
    return cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 2)


def crop_bbox(image, position):
    x, y, w, h = position['x'], position['y'], position['w'], position['h']
    return image[int(x):int(x + w), int(y):int(y + h), :]


def print_str_list(str_list, st_component):
    for strr in str_list:
        st_component.text(strr)
        time.sleep(5)
