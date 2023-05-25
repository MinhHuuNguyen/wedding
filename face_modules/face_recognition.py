from deepface import DeepFace
import cv2

REG_MODEL_NAME = 'ArcFace' # SFace, DeepID, OpenFace, ArcFace
DET_MODEL_NAME = 'opencv' # opencv, ssd

REGISTER_FACE_PATH = 'image/register_minh.jpeg'
# REGISTER_FACE_PATH = 'image/register_thao_anh.jpeg'
REGISTER_FACE = cv2.imread(REGISTER_FACE_PATH)

def recognize_face(face, short_size=None):
    if short_size is not None:
        old_dim = face.shape[:2]
        new_dim = (short_size, int(short_size * old_dim[0] / old_dim[1]))
        face = cv2.resize(face, new_dim, interpolation=cv2.INTER_AREA)

    result = DeepFace.verify(
        img1_path=face,
        img2_path=REGISTER_FACE,
        model_name=REG_MODEL_NAME,
        detector_backend=DET_MODEL_NAME,
        distance_metric='cosine',
        enforce_detection=False,
        align=True,
        normalization='base',
    )

    if short_size is not None:
        # Resize back to original size
        result['facial_areas']['img1']['x'] *= (old_dim[1] / new_dim[0])
        result['facial_areas']['img1']['y'] *= (old_dim[0] / new_dim[1])
        result['facial_areas']['img1']['w'] *= (old_dim[1] / new_dim[0])
        result['facial_areas']['img1']['h'] *= (old_dim[0] / new_dim[1])

    # print(result)
    return result

if __name__ == '__main__':
    print(recognize_face('image/minh_2.jpg'))