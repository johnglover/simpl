import sys
import numpy as np
import scipy.io.wavfile as wav
import simpl

usage = 'Usage: python {0} '.format(__file__) + \
    '<input wav file> <time scale factor> <output wav file>'

if len(sys.argv) != 4:
    print(usage)
    sys.exit(1)

audio, sampling_rate = simpl.read_wav(sys.argv[1])
time_scale_factor = float(sys.argv[2])
output_file = sys.argv[3]

pd = simpl.LorisPeakDetection()
peaks = pd.find_peaks(audio)
pt = simpl.SMSPartialTracking()
partials = pt.find_partials(peaks)

synth = simpl.SndObjSynthesis()
audio_out = np.array([])
step_size = 1.0 / time_scale_factor
current_frame = 0

while current_frame < len(partials):
    i = int(current_frame)
    frame = synth.synth_frame(partials[i])
    audio_out = np.hstack((audio_out, frame))
    current_frame += step_size

audio_out = np.asarray(audio_out * 32768, np.int16)
wav.write(output_file, sampling_rate, audio_out)
