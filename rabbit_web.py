import time
import threading

import cv2
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from face_modules import recognize_face, detect_face, analyze_face
from speech_modules import sound, autoplay_audio
from utils import draw_bbox, crop_bbox, print_str_list


# COMPONENTS
st.set_page_config(layout="wide")
COL_1, COL_2 = st.columns([2, 1])
FRAME_WINDOW = COL_1.image([])
# NEXT_BUTTON = COL_2.button('Tiếp tục...', use_container_width=False)

WEBCAM_CAPTURE = cv2.VideoCapture(0)
WEBCAM_CAPTURE.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
WEBCAM_CAPTURE.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# CONFIGS
NUM_CONFIRMED_FRAMES = 25
SHORT_SIZE = 500


# STATUS
# RUN_ALL = st.checkbox('Run')
RUN_ALL = True
RUN_FACE_RECOGNITION = True
RUN_FACE_ANALYSIS = False
is_shown_reg_result = False

dang_nhap_thanh_cong_str_thread = threading.Thread(
    target=print_str_list,
    args=([
        'Đăng nhập thành công roài.',
        'Xin chào chú Thỏ.',
        'Cơ mà chú Thỏ chưa cười tươi thì phải.',
        'Chú Thỏ vẫn nhìn thẳng vào cam,',
        'bấm nút Tiếp tục trên màn hình,',
        'rồi cười thật tươi vào nhé.'
    ], COL_2))
add_script_run_ctx(dang_nhap_thanh_cong_str_thread)
dang_nhap_thanh_cong_audio_thread = threading.Thread(target=autoplay_audio, args=('dang_nhap_thanh_cong',))
add_script_run_ctx(dang_nhap_thanh_cong_audio_thread)


result_fr_list, result_fa_list = [], []
while RUN_ALL:
    _, frame = WEBCAM_CAPTURE.read()
    if frame is None:
        continue

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
                print('Đăng nhập thành công')
                dang_nhap_thanh_cong_str_thread.start()
                dang_nhap_thanh_cong_audio_thread.start()
                # autoplay_audio('dang_nhap_thanh_cong')
                RUN_FACE_ANALYSIS = True

    # Phase 2: Finish face recognition, run face analysis
    elif not RUN_FACE_RECOGNITION and RUN_FACE_ANALYSIS:
        emotion = analyze_face(crop_bbox(frame, fd_result['facial_area']))
        print(emotion)
        result_fa_list.append(True if emotion == 'happy' else False)

        if len(result_fa_list) >= NUM_CONFIRMED_FRAMES:
            RUN_FACE_ANALYSIS = not all(result_fa_list[-1 * NUM_CONFIRMED_FRAMES:])
            if not RUN_FACE_ANALYSIS:
                COL_2.text(f'Ô kê rồiiiii, tươi rồi...')


if st.button('Record'):
    with st.spinner(f'Recording for {sound.duration} seconds ....'):
        sound.record()
    st.success("Recording completed")
