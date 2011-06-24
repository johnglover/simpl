# Copyright (c) 2009 John Glover, National University of Ireland, Maynooth
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

import simpl
import numpy as np
from scipy.io.wavfile import read, write

input_file = '../tests/audio/flute.wav'
output_file = 'residual.wav'

audio_data = read(input_file)
audio = np.asarray(audio_data[1]) / 32768.0
sampling_rate = audio_data[0]
hop_size = 512
num_frames = len(audio) / hop_size
num_samples = len(audio)
max_peaks = 10
max_partials = 10

pd = simpl.SMSPeakDetection()
pd.max_peaks = max_peaks
pd.hop_size = hop_size
peaks = pd.find_peaks(audio)
pt = simpl.SMSPartialTracking()
pt.max_partials = max_partials
partials = pt.find_partials(peaks)
synth = simpl.SMSSynthesis()
synth.hop_size = hop_size
synth.max_partials = max_partials
synth_audio = synth.synth(partials)
r = simpl.SMSResidual()
r.hop_size = hop_size
audio_out = r.synth(synth_audio, audio)
audio_out = np.asarray(audio_out * 32768, np.int16)
write(output_file, 44100, audio_out)
