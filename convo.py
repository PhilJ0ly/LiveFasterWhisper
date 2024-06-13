# Philippe Joly
# 2024-06-13

# This is a first attempt at creating a conversational AI. This effort is divided into 4 challenges:
# 1. Fast speech to text: using Speech_Recognition for audio capturing and faster-whisper for ASR
# 2. Response generation: using (for now) Llama 3 8b model [ ONLY FOR ENGLISH ]
# 3. Fast text to speech: using
# 4. The connection and timing between the 3 other moving parts.


from dotenv import load_dotenv, find_dotenv
import numpy as np
from queue import Queue
from datetime import datetime, timedelta
from time import sleep
import os

import speech_recognition as sr
from faster_whisper import WhisperModel

from langchain_huggingface import HuggingFaceEndpoint as hf, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

params = {
    "energy": 1000,
    "sample rate": 16000,
    "whisper model": "base",
    "whisper device": ["cpu", "int8"],
    "phrase time limit": 100,
    "llm id": "facebook/blenderbot-1B-distill",
    "model type": "text2text-generation",
    "latency": 2,
    "beams": 5
}


def main():
    load_dotenv(find_dotenv())

    data_queue = Queue()
    phrase_time = None
    transcription = ['[Speaker 1] ']

    try:
        recorder = sr.Recognizer()
        recorder.energy_threshold = params["energy"]
        recorder.dynamic_energy_threshold = False
        source = sr.Microphone(sample_rate=params["sample rate"])
        print("Mic is now hot...")
    except Exception as e:
        print("an error setting up the mic occured: ", str(e))

    try:
        whisk = WhisperModel(params["whisper model"], device=params["whisper device"]
                             [0], compute_type=params["whisper device"][1])

        tokenizer = AutoTokenizer.from_pretrained(params["llm id"])
        model = AutoModelForSeq2SeqLM.from_pretrained(params["llm id"])
        pipe = pipeline(params["model type"], model=model,
                        tokenizer=tokenizer, max_length=100)
        llm = HuggingFacePipeline(pipeline=pipe)
        print("Models loaded and ready to fire")
    except Exception as e:
        print("an error occured loading the models: ", str(e))

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        # at end of recording
        data = audio.get_raw_data()
        data_queue.put(data)

    # background thread passing feed
    recorder.listen_in_background(
        source, record_callback, phrase_time_limit=params["phrase time limit"])

    print("Start the conversation...\n")

    # main conversation loop
    flag = True

    while True:
        try:
            now = datetime.now()

            if phrase_time and flag and now-phrase_time > timedelta(seconds=params["latency"]):
                flag = False
                response = llm.invoke(transcription[-1])
                print("[LLM] ", response)

            # pull audio from queue
            if not data_queue.empty():
                phrase_complete = False
                flag = True

                # if enough time since phrase we consider complete
                if phrase_time and now-phrase_time > timedelta(seconds=params["latency"]):
                    phrase_complete = True

                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                audio_np = np.frombuffer(
                    audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                segments, _ = whisk.transcribe(
                    audio_np, beam_size=params["beams"])

                for seg in segments:
                    if phrase_complete:
                        transcription.append("[Speaker 1] " + seg.text)
                        phrase_complete = False
                    else:
                        transcription[-1] = transcription[-1] + seg.text

                print(transcription[-1])
            else:
                sleep(0.2)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("error occured: ", str(e))
            break

    print("\nKey board interuption\nConversation Over!")


if __name__ == "__main__":
    main()
