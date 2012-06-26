import os
import numpy as np
import scipy.io.wavfile as wavfile
from nose.tools import assert_almost_equals
import simpl.base as base


class TestFrame(object):
    def test_buffers(self):
        N = 256
        f = base.Frame(N)
        assert f.size == N

        a = np.random.rand(N)
        f.audio = a
        assert np.all(f.audio == a)

        a = np.random.rand(N)
        f.synth = a
        assert np.all(f.synth == a)

        a = np.random.rand(N)
        f.residual = a
        assert np.all(f.residual == a)

        a = np.random.rand(N)
        f.synth_residual = a
        assert np.all(f.synth_residual == a)

    def test_peaks(self):
        p = base.Peak()
        p.amplitude = 0.5
        p.frequency = 220.0
        p.phase = 0.0

        f = base.Frame()
        assert f.num_peaks == 0
        assert f.max_peaks > 0
        f.add_peak(p)

        assert f.num_peaks == 1
        assert f.peak(0).amplitude == p.amplitude
        assert f.peaks[0].amplitude == p.amplitude

        f.clear()
        assert f.num_peaks == 0


class TestPeakDetection(object):
    float_precision = 5
    frame_size = 512
    hop_size = 512
    audio_path = os.path.join(
        os.path.dirname(__file__), 'audio/flute.wav'
    )

    @classmethod
    def setup_class(cls):
        cls.audio = wavfile.read(cls.audio_path)[1]
        cls.audio = np.asarray(cls.audio, dtype=np.double)
        cls.audio /= np.max(cls.audio)

    def test_peak_detection(self):
        pd = base.PeakDetection()
        pd.find_peaks(self.audio)

        assert len(pd.frames) == len(self.audio) / self.hop_size
        assert len(pd.frames[0].peaks) == 0
