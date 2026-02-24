from faster_whisper import WhisperModel


MODEL = WhisperModel(
    model_size='large-v2',
    device='cpu',
    compute_type='int8'
)


def translate_speech2text(audio):
    segments, info = MODEL.transcribe('audio.mp3', beam_size=5)

    for segment in segments:
        print('[%.2fs -> %.2fs] %s' % (segment.start, segment.end, segment.text))


if __name__ == '__main__':
    translate_speech2text()
