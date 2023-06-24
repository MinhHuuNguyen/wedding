import cv2
import streamlit as st

from face_modules import recognize_face, detect_face, analyze_face
from speech_modules import sound, autoplay_audio
from utils import draw_bbox, crop_bbox


# COMPONENTS
st.set_page_config(layout="wide")
COL_1, COL_2 = st.columns([3, 1])
FRAME_WINDOW = COL_1.image([])
NEXT_BUTTON = COL_2.button('Tiếp tục...', use_container_width=False)
WEBCAM_CAPTURE = cv2.VideoCapture(0)


# CONFIGS
NUM_CONFIRMED_FRAMES = 25
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

    fd_result = detect_face(frame, short_size=SHORT_SIZE)
    frame = draw_bbox(frame, fd_result['facial_area'])
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Phase 1: Run face recognition
    if RUN_FACE_RECOGNITION:
        verified = recognize_face(crop_bbox(frame, fd_result['facial_area']))
        result_fr_list.append(verified)

        if len(result_fr_list) >= NUM_CONFIRMED_FRAMES:
            RUN_FACE_RECOGNITION = not all(result_fr_list[-1 * NUM_CONFIRMED_FRAMES:])
            if not RUN_FACE_RECOGNITION:
                # autoplay_audio()
                COL_2.text('Đăng nhập thành công')
                COL_2.text('Xin chào chú Thỏ...')
                COL_2.text('Cơ mà chú Thỏ chưa cười tươi...')
                COL_2.text('Chú Thỏ bấm Next rồi cười tươi vào nhé...')

    # Phase 2: Finish face recognition, run face analysis
    elif not RUN_FACE_RECOGNITION and RUN_FACE_ANALYSIS:
        emotion = analyze_face(crop_bbox(frame, fd_result['facial_area']))
        print(emotion)
        result_fa_list.append(True if emotion == 'happy' else False)

        if len(result_fa_list) >= NUM_CONFIRMED_FRAMES:
            RUN_FACE_ANALYSIS = not all(result_fa_list[-1 * NUM_CONFIRMED_FRAMES:])
            if not RUN_FACE_ANALYSIS:
                COL_2.text(f'Ô kê rồiiiii, tươi rồi...')

    # Check result and click of phase 1
    if NEXT_BUTTON:
        if not RUN_FACE_RECOGNITION:
            RUN_FACE_ANALYSIS = True
            NEXT_BUTTON = False


if st.button('Record'):
    with st.spinner(f'Recording for {sound.duration} seconds ....'):
        sound.record()
    st.success("Recording completed")
