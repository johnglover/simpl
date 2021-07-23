import sys
import matplotlib.pyplot as plt
import simpl

usage = 'Usage: python {0} <wav file>'.format(__file__)
if len(sys.argv) != 2:
    print(usage)
    sys.exit(1)

audio = simpl.read_wav(sys.argv[1])[0]

pd = simpl.LorisPeakDetection()
pd.max_peaks = 30
frames = pd.find_peaks(audio)
pt = simpl.MQPartialTracking()
pt.max_partials = 30
frames = pt.find_partials(frames)
simpl.plot_partials(frames, show_peaks=False)
plt.show()
