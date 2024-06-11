# Philippe Joly
# 2024-06-10
# Implementation of faster-whisper for live speech to text transcription

from faster_whisper import WhisperModel
import numpy as np
import speech_recognition as sr
# from pydub import AudioSegment

from queue import Queue
from time import sleep
from datetime import datetime, timedelta
import os


class model():
    def __init__(self, model_size, energy=1000, latency=2, phrase_timeout=3, dev='cuda', cmp_type='int8'):
        self.model = WhisperModel(
            model_size, device=dev, compute_type=cmp_type)
        self.energy = energy
        self.latency = latency
        self.phrase_timeout = phrase_timeout

    async def transcribe(self, audio, beam_size=5):
        await self.model.transcribe(audio, beam_size=beam_size)


def main():

    # last time recording was taken from queue
    phrase_time = None

    data_queue = Queue()

    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False
    source = sr.Microphone(sample_rate=16000)

    # whisk = model(model_size="base", device="cpu", cmp_type="int8")
    whisk = WhisperModel('base', device='cpu', compute_type="int8")

    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        # what happens on end of recording
        data = audio.get_raw_data()
        data_queue.put(data)

    # background thread passingf raw audio bytes
    recorder.listen_in_background(
        source, record_callback, phrase_time_limit=2)

    print("Model Loaded!!!!\n")

    while True:
        try:
            now = datetime.now()

            # pull audio from queue
            if not data_queue.empty():
                phrase_complete = False

                # if enough time b/w recordings we consider the phrase complete
                # we then clear the current working audio buffer to start over with new data
                if phrase_time and now - phrase_time > timedelta(seconds=3):
                    phrase_complete = True

                phrase_time = now

                # combine audio from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                audio_np = np.frombuffer(
                    audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                # print(audio_np.shape)

                # audio_seg = AudioSegment(
                #     audio_data,
                #     sample_width=2,
                #     frame_rate=16000,
                #     channels=1
                # )
                # audio_seg.export('t.mp3', format="mp3")

                segments, info = whisk.transcribe(audio_np, beam_size=5)

                text = ''
                for segment in segments:
                    text = text + segment.text + ' '

                # if phrase_complete:
                transcription.append(text)
                # else:
                # transcription[-1] = text

                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                    print('', end='', flush=True)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
