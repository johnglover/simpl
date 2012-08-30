import sys
import numpy as np
import scipy.io.wavfile as wav
import simpl

usage = 'Usage: python {0} <input wav file> <output wav file>'.format(__file__)
if len(sys.argv) != 3:
    print usage
    sys.exit(1)

audio = simpl.read_wav(sys.argv[1])[0]
output_file = sys.argv[2]

pd = simpl.LorisPeakDetection()
peaks = pd.find_peaks(audio)
pt = simpl.SMSPartialTracking()
partials = pt.find_partials(peaks)
synth = simpl.LorisSynthesis()
audio_out = synth.synth(partials)
audio_out = np.asarray(audio_out * 32768, np.int16)
wav.write(output_file, 44100, audio_out)
