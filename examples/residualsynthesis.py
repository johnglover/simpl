import sys
import numpy as np
import scipy.io.wavfile as wav
import simpl

usage = 'Usage: python {0} <input wav file> <output wav file>'.format(__file__)
if len(sys.argv) != 3:
    print(usage)
    sys.exit(1)

audio, sampling_rate = simpl.read_wav(sys.argv[1])
output_file = sys.argv[2]

r = simpl.SMSResidual()
audio_out = r.synth(audio)
audio_out = np.asarray(audio_out * 32768, np.int16)
wav.write(output_file, sampling_rate, audio_out)
