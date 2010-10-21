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
output_file = 'resynth.wav'

audio_data = read(input_file)
audio = simpl.asarray(audio_data[1]) / 32768.0  # values between -1 and 1
sample_rate = audio_data[0]

pd = simpl.SMSPeakDetection()
pd.max_peaks = 40
pd.hop_size = 512
peaks = pd.find_peaks(audio)
pt = simpl.MQPartialTracking()
pt.max_partials = 20
partials = pt.find_partials(peaks)
sndobj_synth = simpl.SndObjSynthesis()

audio_out = sndobj_synth.synth(partials)
audio_out = np.asarray(audio_out * 32768, np.int16)
write(output_file, 44100, audio_out)
