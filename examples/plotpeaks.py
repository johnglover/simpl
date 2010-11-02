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
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

input_file = '../tests/audio/flute.wav'
audio_in_data = read(input_file)
audio_in = simpl.asarray(audio_in_data[1]) / 32768.0  # values between -1 and 1
sample_rate = audio_in_data[0]

# take just the first few frames
audio = audio_in[0:4096]
# peak detection using the SndObj library
pd = simpl.SndObjPeakDetection()
pd.max_peaks = 20
peaks = pd.find_peaks(audio)
# plot peaks using matplotlib
simpl.plot.plot_peaks(peaks)
plt.show()

