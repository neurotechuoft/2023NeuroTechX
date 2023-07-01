import RPi.GPIO as io
import time
import tensorflow as tf
from tensorflow import keras
import pyOpenBCI
import mne

#setup for GPIO pins; 4 set as output to buzzer, 5 and 6 as EOG input channels
io.setmode(io.BCM)
io.setup(4, io.OUT)
io.setup(5, io.IN)
io.setup(6, io.IN)

#open and save tensorflow lite model
with open("tensorflow_lite_model.tflite", 'rb') as fid:
    tflite_model = fid.read()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

#connect to OpenBCI Cyton board with bluetooth dongle and let the library automatically find the port the dongle is connected to
board = pyOpenBCI.OpenBCICyton(port=None)

def data_callback(sample):
    #sample is the eeg values; we take the first two channels and discard the rest
    eeg_data = sample.channels_data[:2]
    #digital eog signals are taken from the arduino
    eog_data1 = GPIO.input(5)
    eog_data2 = GPIO.input(6)
    
    #filtering the data with online bandpass filters
    raw_eeg1 = mne.io.read_raw_brainvision(eeg_data[0])
    raw_eeg2 = mne.io.read_raw_brainvision(eeg_data[1])
    raw_eog1 = mne.io.read_raw_brainvision(eog_data1)
    raw_eog2 = mne.io.read_raw_brainvision(eog_data2)

    low_cuteeg = 0.1
    hi_cuteeg = 30
    low_cuteog = 0.1
    hi_cuteog = 15

    filt_eeg1 = raw_eeg1.copy().filter(low_cuteeg, high_cuteeg)
    filt_eeg2 = raw_eeg2.copy().filter(low_cuteeg, high_cuteeg)
    filt_eog1 = raw_eog1.copy().filter(low_cuteog, high_cuteog)
    filt_eog2 = raw_eog2.copy().filter(low_cuteog, high_cuteog)

    #data is combined into the input format expected by the model
    combined_data = [filt_eeg1, filt_eeg2, filt_eog1, filt_eog2]

    #model called with the data
    my_signature = interpreter.get_signature_runner()
    output = my_signature(x=combined_data, shape=(1, 4, 1), dtype=tf.float32))
    
    #if the model predicts anything but an awake state, sound the buzzer. Otherwise, do not sound the buzzer.
    if(output != 00):
        io.output(4, 1)
    else:
        io.output(4, 0)

#starting datastream from board; this callback function will be called continuously with the eeg values
board.start_stream(data_callback)
