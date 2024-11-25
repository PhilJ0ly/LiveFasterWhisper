# LiveFasterWhisper
This is a simple implementation of new incredibly fast text to speech whisper model. This is a simple conversation agent consiting of 3 parts: speech recognition using the **speech_recognition** library, text generation using **Lamma** model by Facebook, and **faster_whisper** for the text-to-speech. One drawback is that, without a GPU the latency is not satisfactory.
