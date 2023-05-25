import cv2
import streamlit as st

from face_modules import recognize_face, detect_face, analyze_face
from utils import draw_bbox


# COMPONENTS
st.set_page_config(layout="wide")
COL_1, COL_2 = st.columns([3, 1])
FRAME_WINDOW = COL_1.image([])
NEXT_BUTTON = COL_2.button('Tiếp tục...', use_container_width=False)
WEBCAM_CAPTURE = cv2.VideoCapture(0)


# CONFIGS
NUM_CONFIRMED_FRAMES = 1
SHORT_SIZE = 500


# STATUS
RUN_ALL = COL_2.checkbox('Run')
RUN_FACE_RECOGNITION = True
RUN_FACE_ANALYSIS = False
is_shown_reg_result = False


result_fr_list, result_fa_list = [], []
while RUN_ALL:
    _, frame = WEBCAM_CAPTURE.read()
    frame = cv2.flip(frame, 1) # horizontal flip

    # Phase 1: Run face recognition
    if RUN_FACE_RECOGNITION:
        result = recognize_face(frame, short_size=SHORT_SIZE)
        result_fr_list.append(result['verified'])

        if len(result_fr_list) >= NUM_CONFIRMED_FRAMES:
            RUN_FACE_RECOGNITION = not all(result_fr_list[-1 * NUM_CONFIRMED_FRAMES:])
            if not RUN_FACE_RECOGNITION:
                COL_2.text('Đăng nhập thành công')
                COL_2.text('Xin chào chú Thỏ...')
                COL_2.text('Cơ mà chú Thỏ chưa cười tươi...')
                COL_2.text('Chú Thỏ bấm Next rồi cười tươi vào nhé...')

        frame = draw_bbox(frame, result['facial_areas']['img1'])

    # Phase 2: Finish face recognition, run face analysis
    elif not RUN_FACE_RECOGNITION and RUN_FACE_ANALYSIS:
        result = analyze_face(frame, short_size=SHORT_SIZE)
        result_fa_list.append(True if result['dominant_emotion'] == 'happy' else False)

        if len(result_fa_list) >= NUM_CONFIRMED_FRAMES:
            RUN_FACE_ANALYSIS = not all(result_fa_list[-1 * NUM_CONFIRMED_FRAMES:])
            if not RUN_FACE_ANALYSIS:
                print(result_fa_list[-1 * NUM_CONFIRMED_FRAMES:])
                COL_2.text(f'Ô kê rồiiiii, tươi rồi...')

        frame = draw_bbox(frame, result['region'])

    # Phase 3: Finish face recognition and face analysis, run face detection only
    else:
        result = detect_face(frame, short_size=SHORT_SIZE)
        frame = draw_bbox(frame, result['facial_area'])

    # Check result and click of phase 1
    if NEXT_BUTTON:
        if not RUN_FACE_RECOGNITION:
            RUN_FACE_ANALYSIS = True
            NEXT_BUTTON = False

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
