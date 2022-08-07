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
from flask import Flask, render_template, request, session, redirect, url_for

app = Flask(__name__)

model = keras.models.load_model('models/language-detection.h5')

app.secret_key='test'

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
    max = 0
    for spectrogram, label in ds.batch(1):
        prediction = model(spectrogram)
        class_predictions = tf.nn.softmax(prediction[0])
        for i, c in enumerate(classes):
            if class_predictions[i].numpy() > max:
                max = class_predictions[i].numpy()
                max_class = c
    result = max_class
    result = str(result)

    #result = str(save_path)
    return render_template("input.html",result = result)

#if __name__ == '__main__':
    #app.run()
