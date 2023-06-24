import cv2
import numpy as np
from deepface import DeepFace
from deepface.extendedmodels.Emotion import *

DET_MODEL_NAME = 'opencv' # opencv, ssd


EMOTION_MODEL = DeepFace.build_model('Emotion')


def analyze_face(face_img):
    img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (48, 48))
    img_gray = np.expand_dims(img_gray, axis=0)

    emotion_predictions = EMOTION_MODEL.predict(img_gray, verbose=0)[0]

    # return Emotion.labels[np.argmax(emotion_predictions)]
    return np.argmax(emotion_predictions)
