import simpl
import numpy as np
from scipy.io.wavfile import read, write

input_file = '../../tests/audio/flute.wav'
output_file = 'residual.wav'

audio_data = read(input_file)
audio = np.asarray(audio_data[1]) / 32768.0
sampling_rate = audio_data[0]
hop_size = 512

r = simpl.SMSResidual()
r.hop_size = hop_size
audio_out = r.synth(audio)
audio_out = np.asarray(audio_out * 32768, np.int16)
write(output_file, 44100, audio_out)
