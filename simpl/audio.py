import numpy as np
import scipy.io.wavfile as wav
import simpl


def read_wav(file):
    'return floating point values between -1 and 1'
    sampling_rate, audio = wav.read(file)

    # if wav file has more than 1 channel, just take the first one
    if audio.ndim > 1:
        audio = audio.T[0]

    return np.asarray(audio, dtype=simpl.dtype) / 32768.0, sampling_rate
