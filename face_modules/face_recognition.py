import cv2
import numpy as np
from deepface import DeepFace
from deepface.commons import functions, distance as distance_fn


# REGISTER_FACE_PATH = 'image/register_thao_anh.jpeg'
REGISTER_FACE_PATH = 'image/register_minh.jpeg'


REG_MODEL_NAME = 'ArcFace' # SFace, DeepID, OpenFace, ArcFace
FR_MODEL = DeepFace.build_model(REG_MODEL_NAME)
FR_IMG_TARGET_SIZE = functions.find_target_size(REG_MODEL_NAME)
FR_EMB_THRESHOLD = distance_fn.findThreshold(REG_MODEL_NAME, 'cosine')


def get_emb(face_img):
    face_img = cv2.resize(face_img, FR_IMG_TARGET_SIZE)
    face_img = np.expand_dims(face_img, axis=0)
    face_img = functions.normalize_input(img=face_img, normalization='base')
    face_embedding = FR_MODEL.predict(face_img, verbose=0)[0]

    return face_embedding


REGISTER_FACE_EMB = get_emb(cv2.imread(REGISTER_FACE_PATH))


def recognize_face(face_img):
    if face_img is not None:
        checking_emb = get_emb(face_img)
        distance = distance_fn.findCosineDistance(REGISTER_FACE_EMB, checking_emb)

        return distance <= FR_EMB_THRESHOLD

    return False
