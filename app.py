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
from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory

app = Flask(__name__)

model = keras.models.load_model('models/language-detection.h5')

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
        audio_file = request.files['recorder']
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

    #result = str(save_path)
    return render_template("input.html",result = result, second_result = second_result, third_result = third_result, finish=finish)

#if __name__ == '__main__':
    #app.run()
