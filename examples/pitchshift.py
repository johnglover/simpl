import sys
import math
import numpy as np
import scipy.io.wavfile as wav
import simpl

usage = 'Usage: python {0} '.format(__file__) + \
    '<input wav file> <pitch shift amount> <output wav file>'

if len(sys.argv) != 4:
    print(usage)
    sys.exit(1)

audio, sampling_rate = simpl.read_wav(sys.argv[1])
pitch_shift_amount = float(sys.argv[2])
output_file = sys.argv[3]

pd = simpl.LorisPeakDetection()
frames = pd.find_peaks(audio)
pt = simpl.SMSPartialTracking()
frames = pt.find_partials(frames)

twelfth_root_2 = math.pow(2.0, 1.0 / 12)
freq_scale = math.pow(twelfth_root_2, pitch_shift_amount)

for frame in frames:
    partials = frame.partials
    for p in partials:
        p.frequency *= freq_scale
    frame.partials = partials

synth = simpl.SndObjSynthesis()
harm_synth = synth.synth(frames)
r = simpl.SMSResidual()
res_synth = r.synth(audio)

audio_out = harm_synth + res_synth
audio_out = np.asarray(audio_out * 32768, np.int16)
wav.write(output_file, sampling_rate, audio_out)
