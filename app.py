# Philippe Joly
# 2024-06-10
# Implementation of faster-whisper text transcription

from faster_whisper import WhisperModel

audio = "audio.m4a"

model_size = "large-v3"

model = WhisperModel(model_size, device='cpu', compute_type="int8")

segments, info = model.transcribe(audio, beam_size=5)

print("Detected language '%s' with probability %f" %
      (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
