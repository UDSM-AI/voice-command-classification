import numpy as np

from tensorflow.keras import models

from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer
from fingerprint import ExportModel, label_names, load_model

#  Modify this in the correct order
commands = label_names
loaded_model  = ExportModel(load_model())

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    results = loaded_model(spec)

    predicted_class = results['class_names'].numpy()[0].decode('utf-8')
    print(predicted_class)

if __name__ == "__main__":
    # from turtle_helper import move_turtle
    while True:
        command = predict_mic()
        # move_turtle(command)
        # if command == "stop":
        #     terminate()
        #     break