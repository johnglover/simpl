import simpl
import numpy as np
from scipy.io.wavfile import read, write

input_file = '../../tests/audio/flute.wav'
output_file = 'residual.wav'

audio_data = read(input_file)
audio = np.asarray(audio_data[1]) / 32768.0
sampling_rate = audio_data[0]
hop_size = 512
num_frames = len(audio) / hop_size
num_samples = len(audio)
max_peaks = 10
max_partials = 10

pd = simpl.SMSPeakDetection()
pd.max_peaks = max_peaks
pd.hop_size = hop_size
peaks = pd.find_peaks(audio)
pt = simpl.SMSPartialTracking()
pt.max_partials = max_partials
partials = pt.find_partials(peaks)
synth = simpl.SMSSynthesis()
synth.hop_size = hop_size
synth.max_partials = max_partials
synth_audio = synth.synth(partials)
r = simpl.SMSResidual()
r.hop_size = hop_size
audio_out = r.synth(synth_audio, audio)
audio_out = np.asarray(audio_out * 32768, np.int16)
write(output_file, 44100, audio_out)
