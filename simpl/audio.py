import numpy as np
from scipy.io.wavfile import read, write
import simpl

def read_wav(file):
    audio_data = read(file)
    # return floating point values between -1 and 1
    return simpl.asarray(audio_data[1]) / 32768.0, audio_data[0]

