# Copyright (c) 2010 John Glover, National University of Ireland, Maynooth
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
from simpl import lp
import numpy as np
import matplotlib.pyplot as plt
import unittest

audio, sampling_rate = simpl.read_wav("audio/flute.wav")

# take just the first few frames
audio = audio_in[0:4096]
# Peak detection and partial tracking using SMS
pd = simpl.SndObjPeakDetection()
pd.max_peaks = 60
peaks = pd.find_peaks(audio)
pt = simpl.MQPartialTracking()
pt.max_partials = 60
partials = pt.find_partials(peaks)
simpl.plot.plot_partials(partials)
plt.show()

exit()


class TestLP(unittest.TestCase):
    FLOAT_PRECISION = 3 # number of decimal places to check for accuracy

    def test_predict(self):
        """test_predict"""
        coefs = np.array([1,2,3,4,5])
        test_signal = np.ones(5)
        predictions = lp.predict(test_signal, coefs, 2)
        self.assertEquals(predictions[0], -sum(coefs))
        self.assertEquals(predictions[1], -sum(coefs[1:])-predictions[0])
    
suite = unittest.TestSuite()
suite.addTest(TestLP('test_predict'))
unittest.TextTestRunner().run(suite)

