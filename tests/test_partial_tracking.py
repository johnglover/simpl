import os
import numpy as np
from nose.tools import assert_almost_equals
import simpl
import simpl.peak_detection as peak_detection
import simpl.partial_tracking as partial_tracking

float_precision = 5
frame_size = 512
hop_size = 512
audio_path = os.path.join(
    os.path.dirname(__file__), 'audio/flute.wav'
)


class TestPartialTracking(object):
    @classmethod
    def setup_class(cls):
        cls.audio = simpl.read_wav(audio_path)[0]

    def test_partial_tracking(self):
        pd = peak_detection.PeakDetection()
        frames = pd.find_peaks(self.audio)

        pt = partial_tracking.PartialTracking()
        frames = pt.find_partials(frames)

        assert len(frames) == len(self.audio) / hop_size
        assert len(frames[0].partials) == 100