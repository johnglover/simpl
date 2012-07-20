import matplotlib.pyplot as plt
import simpl

input_file = '../../tests/audio/flute.wav'
audio = simpl.read_wav(input_file)[0]
audio = audio[len(audio) / 2:(len(audio) / 2) + 4096]

pd = simpl.SndObjPeakDetection()
pd.max_peaks = 60
peaks = pd.find_peaks(audio)
pt = simpl.MQPartialTracking()
pt.max_partials = 60
partials = pt.find_partials(peaks)
simpl.plot_partials(partials)
plt.show()
