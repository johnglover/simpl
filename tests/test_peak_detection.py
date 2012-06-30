import os
import numpy as np
from nose.tools import assert_almost_equals
import simpl
import simpl.peak_detection as peak_detection

float_precision = 5
frame_size = 512
hop_size = 512
audio_path = os.path.join(
    os.path.dirname(__file__), 'audio/flute.wav'
)


class TestPeakDetection(object):
    @classmethod
    def setup_class(cls):
        cls.audio = simpl.read_wav(audio_path)[0]

    def test_peak_detection(self):
        pd = peak_detection.PeakDetection()
        pd.find_peaks(self.audio)

        assert len(pd.frames) == len(self.audio) / hop_size
        assert len(pd.frames[0].peaks) == 0
