import simpl
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

input_file = '../../tests/audio/flute.wav'
audio_data = read(input_file)
audio = simpl.asarray(audio_data[1]) / 32768.0  # values between -1 and 1
sample_rate = audio_data[0]

# take just the first few frames
audio = audio[0:4096]
# peak detection using the SndObj library
pd = simpl.SndObjPeakDetection()
pd.max_peaks = 20
peaks = pd.find_peaks(audio)
# plot peaks using matplotlib
simpl.plot.plot_peaks(peaks)
plt.show()
