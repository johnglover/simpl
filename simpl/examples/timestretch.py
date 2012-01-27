import simpl
from simpl.fx import time_stretch
from scipy.io.wavfile import read, write
import numpy as np

input_file = '../../tests/audio/flute.wav'
output_file = 'flute_2x.wav'
time_stretch_factor = 2

audio_in_data = read(input_file)
audio_in = simpl.asarray(audio_in_data[1]) / 32768.0  # values between -1 and 1
sample_rate = audio_in_data[0]

print "Time stretching", input_file, "by a factor of", time_stretch_factor
pd = simpl.SndObjPeakDetection()
pd.max_peaks = 100
peaks = pd.find_peaks(audio_in)
pt = simpl.SndObjPartialTracking()
pt.max_partials = 10
partials = pt.find_partials(peaks)
partials = time_stretch(partials, time_stretch_factor)
sndobj_synth = simpl.SndObjSynthesis()
audio_out = sndobj_synth.synth(partials)
audio_out = np.asarray(audio_out * 32768, np.int16)
print "Writing output to", output_file
write(output_file, 44100, audio_out)
