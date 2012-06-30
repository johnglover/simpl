import os
import numpy as np
from nose.tools import assert_almost_equals
import simpl
import simpl.peak_detection as peak_detection
import simpl.partial_tracking as partial_tracking
import simpl.synthesis as synthesis
import simpl.residual as residual

float_precision = 5
frame_size = 512
hop_size = 512
audio_path = os.path.join(
    os.path.dirname(__file__), 'audio/flute.wav'
)


class TestResidual(object):
    @classmethod
    def setup_class(cls):
        cls.audio = simpl.read_wav(audio_path)[0]

    def test_synthesis(self):
        pd = peak_detection.PeakDetection()
        frames = pd.find_peaks(self.audio)

        pt = partial_tracking.PartialTracking()
        frames = pt.find_partials(frames)

        s = synthesis.Synthesis()
        synth_audio = s.synth(frames)

        r = residual.Residual()
        residual_audio = r.find_residual(synth_audio, self.audio)
        assert len(residual_audio)
