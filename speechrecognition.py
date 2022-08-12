import speech_recognition as sr

speech_engine = sr.Recognizer()

languages = ['it', 'fr', 'de', 'en']

for i in range(1, 10):
    path = "C:/Users/bened/notebooks/ML/example_voices/Italian/Audiospur-" + str(i) + ".wav"
    path = "C:/Users/bened/notebooks/ML/example_voices/German/" + str(i) + ".wav"
    with sr.WavFile(path) as source:  # use "test.wav" as the audio source
        audio = speech_engine.record(source)
        max_conf = 0
        for ln in languages:
            try:
                conf = speech_engine.recognize_google(audio, language=ln)['alternative'][0]['confidence']
            except:
                conf = 0
            if conf>max_conf:
                max_conf = conf
                max_ln = ln

    print(max_ln)
# print(confidence)                     # extract audio data from the file
# generate a list of possible transcriptions