import os
import sys
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from myfuncs import *
from flask import Flask, render_template, request, session

app = Flask(__name__)

model = keras.models.load_model('models/language-detection.h5')

app.secret_key='test'

@app.route('/')
def my_form():
    return render_template('input.html')

@app.route('/', methods=['POST'])

def output():
    #import tensorflow as tf

    audio_file = request.form['filename']
    classes = ['Italian' 'French' 'German']
    AUTOTUNE = tf.data.AUTOTUNE

    ds = preprocess_dataset_prediction([str(audio_file)])
    for spectrogram, label in sample_ds.batch(1):
        prediction = model(spectrogram)
        class_predictions = tf.nn.softmax(prediction[0])
        for i, c in enumerate(classes):
            if class_predictions[i].numpy() > max:
                max = class_predictions[i].numpy()
                max_class = c
    result = max_class
    result = str(result)
    
    return render_template("input.html",result = result)

#if __name__ == '__main__':
    #app.run()
