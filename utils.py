import cv2


def draw_bbox(image, position):
    x, y, w, h = position['x'], position['y'], position['w'], position['h']
    return cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 2)
