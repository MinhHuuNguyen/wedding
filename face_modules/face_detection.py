from deepface import DeepFace
import cv2


DET_MODEL_NAME = 'opencv' # opencv, ssd


def detect_face(face, short_size=None):
    if short_size is not None:
        old_dim = face.shape[:2]
        new_dim = (short_size, int(short_size * old_dim[0] / old_dim[1]))
        face = cv2.resize(face, new_dim, interpolation=cv2.INTER_AREA)

    result = DeepFace.extract_faces(
        img_path=face, 
        target_size=(224, 224), 
        detector_backend=DET_MODEL_NAME,
        enforce_detection=False
    )[0]

    if short_size is not None:
        # Resize back to original size
        result['facial_area']['x'] *= (old_dim[1] / new_dim[0])
        result['facial_area']['y'] *= (old_dim[0] / new_dim[1])
        result['facial_area']['w'] *= (old_dim[1] / new_dim[0])
        result['facial_area']['h'] *= (old_dim[0] / new_dim[1])

    return result
