import base64
from glob import glob

import streamlit as st


def encode_audio_files(path):
    with open(path, "rb") as f:
        data = f.read()
    base64_audio_str = base64.b64encode(data).decode()

    return f'''
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{base64_audio_str}" type="audio/mp3">
        </audio>
    '''


AUDIO_PATHS = glob('audio/*.wav')
MD_STR_DICT = {}
for path in AUDIO_PATHS:
    name = path.split('/')[-1][:-4]
    MD_STR_DICT[name] = encode_audio_files(path)


def autoplay_audio(key):
    st.markdown(MD_STR_DICT[key], unsafe_allow_html=True)
