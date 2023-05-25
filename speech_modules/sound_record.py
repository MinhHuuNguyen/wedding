import logging
import pyaudio, wave

logger = logging.getLogger('src.sound')

class Sound(object):
    def __init__(self):
        # Set default configurations for recording device
        self.format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 44100
        self.chunk = 1024
        self.duration = 3
        self.path = 'recorded.wav'
        self.device = 0
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.device_info()
        print()
        logger.info("Audio device configurations currently used")
        logger.info(f"Default input device index = {self.device}")
        logger.info(f"Max input channels = {self.channels}")
        logger.info(f"Default samplerate = {self.sample_rate}")

    def device_info(self):
        num_devices = self.audio.get_device_count()
        keys = ['name', 'index', 'maxInputChannels', 'defaultSampleRate']
        logger.info(f"List of System's Audio Devices configurations:")
        logger.info(f"Number of audio devices: {num_devices}")
        for i in range(num_devices):
            info_dict = self.audio.get_device_info_by_index(i)
            logger.info([(key, value) for key, value in info_dict.items() if key in keys])

    def record(self):
        # Start recording
        self.audio = pyaudio.PyAudio()
        stream = self.audio.open(
                        format=self.format,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk,
                        input_device_index=self.device)
        logger.info(f"Recording started for {self.duration} seconds")
        self.frames = []
        for i in range(0, int(self.sample_rate / self.chunk * self.duration)):
            data = stream.read(self.chunk)
            self.frames.append(data)
        logger.info ("Recording Completed")

        # Stop recording
        stream.stop_stream()
        stream.close()
        self.audio.terminate()
        self.save()

    def save(self):
        waveFile = wave.open(self.path, 'wb')
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.audio.get_sample_size(self.format))
        waveFile.setframerate(self.sample_rate)
        waveFile.writeframes(b''.join(self.frames))
        waveFile.close()
        logger.info(f"Recording saved to {self.path}")

sound = Sound()
