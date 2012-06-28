import simpl
import matplotlib.pyplot as plt

input_file = '../../tests/audio/flute.wav'
audio = simpl.read_wav(input_file)[0]

# take just the first few frames
audio = audio[0:4096]

# peak detection using the SndObj library
pd = simpl.SndObjPeakDetection()
pd.max_peaks = 20
peaks = pd.find_peaks(audio)

# plot peaks using matplotlib
simpl.plot.plot_peaks(peaks)
plt.show()
