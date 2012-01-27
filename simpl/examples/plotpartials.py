import simpl
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

input_file = '../../tests/audio/flute.wav'
audio_in_data = read(input_file)
audio_in = simpl.asarray(audio_in_data[1]) / 32768.0  # values between -1 and 1
sample_rate = audio_in_data[0]

# take just the first few frames
audio = audio_in[0:4096]
# Peak detection and partial tracking using SMS
pd = simpl.SndObjPeakDetection()
pd.max_peaks = 60
peaks = pd.find_peaks(audio)
pt = simpl.MQPartialTracking()
pt.max_partials = 60
partials = pt.find_partials(peaks)
simpl.plot.plot_partials(partials)
plt.show()
