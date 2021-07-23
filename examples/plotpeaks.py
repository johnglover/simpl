import sys
import simpl
import matplotlib.pyplot as plt

usage = 'Usage: python {0} <wav file>'.format(__file__)
if len(sys.argv) != 2:
    print(usage)
    sys.exit(1)

audio = simpl.read_wav(sys.argv[1])[0]

# take just a few frames
audio = audio[len(audio) / 2:(len(audio) / 2) + 4096]

# peak detection using the SndObj library
pd = simpl.SndObjPeakDetection()
pd.max_peaks = 20
frames = pd.find_peaks(audio)

# plot peaks using matplotlib
simpl.plot.plot_peaks(frames)
plt.show()
