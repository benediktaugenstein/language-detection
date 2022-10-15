import os
import sys
import pathlib

import whisper

import numpy as np

import transformers

import ffmpeg

import numpy as np

import torch
import pandas as pd
import torchaudio

from tqdm.notebook import tqdm
from IPython import display

from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory

app = Flask(__name__)

app.secret_key='test'

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def my_form():
    return render_template('input.html')

@app.route('/', methods=['GET', 'POST'])

def output():

    if request.method == 'POST':

        save_path = os.path.join("temp.wav")
        request.files['recorder'].save(save_path)
        #audio_file = request.files['recorder']
    """
    with sr.WavFile(save_path) as source:  # use "test.wav" as the audio source

        audio = speech_engine.record(source)
        max_conf = 0

        for ln in languages:
            try:

                result = speech_engine.recognize_google(audio, language=ln)['alternative'][0]
                conf = result['confidence']
                transcript = result['transcript']

                letters = transcript.replace(' ', '')
                letter_count = len(letters)
                word_count = len(transcript.split())

                conf += 0.0065 * letter_count + 0.005 * word_count

            except:

                conf = 0

            if conf>max_conf:

                max_conf = conf
                max_ln = ln


    
    classes = ['Italian', 'French', 'German']
    AUTOTUNE = tf.data.AUTOTUNE

    ds = preprocess_dataset_prediction([str(save_path)])
    max = -1
    second = 0
    third = 0
    max_class = ''
    second_class = ''
    third_class = ''
    probs = []
    for spectrogram, label in ds.batch(1):
        prediction = model(spectrogram)
        class_predictions = tf.nn.softmax(prediction[0])
        for i, c in enumerate(classes):
            probs.append(class_predictions[i].numpy())
        preds = zip(probs, classes)
        preds2 = sorted(preds, key=lambda x: x[0], reverse=True)

    result = '1. ' + preds2[0][1] + ' - Confidence: ' + str(round(preds2[0][0]*100, 2)) + '%'
    result = str(result)
    second_result = '2. ' + preds2[1][1] + ' - Confidence: ' + str(round(preds2[1][0]*100, 2)) + '%'
    third_result = '3. ' + preds2[2][1] + ' - Confidence: ' + str(round(preds2[2][0]*100, 2)) + '%'
    
    finish = 'finished'
    #result = str(preds)
    #result = str(save_path)
    if max_ln=='':
        lang_result = 'Sorry, language could not be detected.'
    else:
        lang_dict = {'it':'Italian', 'fr':'French', 'de':'German', 'en':'English'}
        lang_result = 'Detected language: ' + lang_dict[max_ln]
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Set Runtime to GPU in Google Colab

    model = whisper.load_model("base")

    transcription = model.transcribe(save_path)

    lang_result = 'Detected Language: ' + transcription['language']
    text = 'Transcription: ' + transcription['text']

    finish = 'finished'
    result = str(lang_result)
    second_result = str(text)
    return render_template("input.html",result = result, second_result=second_result, finish=finish) #second_result = second_result, third_result = third_result,

#if __name__ == '__main__':
    #app.run()
