from deepface import DeepFace
import cv2

DET_MODEL_NAME = 'opencv' # opencv, ssd

def analyze_face(face, short_size=None):
    if short_size is not None:
        old_dim = face.shape[:2]
        new_dim = (short_size, int(short_size * old_dim[0] / old_dim[1]))
        face = cv2.resize(face, new_dim, interpolation=cv2.INTER_AREA)

    result = DeepFace.analyze(
        img_path=face, 
        actions=['emotion'],
        enforce_detection=False,
        detector_backend="opencv",
        silent=True
    )[0]

    if short_size is not None:
        # Resize back to original size
        result['region']['x'] *= (old_dim[1] / new_dim[0])
        result['region']['y'] *= (old_dim[0] / new_dim[1])
        result['region']['w'] *= (old_dim[1] / new_dim[0])
        result['region']['h'] *= (old_dim[0] / new_dim[1])
    return result


if __name__ == '__main__':
    print(analyze_face('image/minh_2.jpg'))
