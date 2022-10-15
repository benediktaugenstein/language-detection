import os
import sys

import numpy as np
import pandas as pd

import whisper
import transformers
import ffmpeg
import torch
import torchaudio
from tqdm.notebook import tqdm

from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory

app = Flask(__name__)

model = whisper.load_model("tiny")
#model = torch.load('base2.pt')
#model.eval()

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

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Set Runtime to GPU in Google Colab

    transcription = model.transcribe(save_path)

    lang_result = 'Detected Language: ' + transcription['language']
    text = 'Transcription: ' + transcription['text']

    finish = 'finished'
    result = str(lang_result)
    second_result = str(text)
    return render_template("input.html",result = result, second_result=second_result, finish=finish) #second_result = second_result, third_result = third_result,

#if __name__ == '__main__':
    #app.run()
