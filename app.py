import os
import sys
import os
import pathlib

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow import keras
from IPython import display
from myfuncs import *
import speech_recognition as sr
from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory

app = Flask(__name__)

model = keras.models.load_model('models/language-detection.h5')

speech_engine = sr.Recognizer()

languages = ['it', 'fr', 'de', 'en']

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


    """
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
    """
    finish = 'finished'
    #result = str(preds)
    #result = str(save_path)
    if max_ln=='':
        lang_result = 'Sorry, language could not be detected.'
    else:
        lang_dict = {'it':'Italian', 'fr':'French', 'de':'German', 'en':'English'}
        lang_result = 'Detected language: ' + lang_dict[max_ln]
    result = str(lang_result)
    return render_template("input.html",result = result, finish=finish) #second_result = second_result, third_result = third_result,

#if __name__ == '__main__':
    #app.run()
