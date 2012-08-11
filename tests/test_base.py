import os
import numpy as np
from nose.tools import assert_almost_equals
import simpl.base as base

float_precision = 5
frame_size = 512
hop_size = 512
audio_path = os.path.join(
    os.path.dirname(__file__), 'audio/flute.wav'
)


class TestFrame(object):
    def test_buffers(self):
        N = 256
        f = base.Frame(N)
        f.synth_size = N
        assert f.size == N
        assert f.synth_size == N

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
        assert len(f.peaks) == 0
        assert f.max_peaks > 0
        f.peaks = [p]

        assert_almost_equals(f.peaks[0].amplitude, p.amplitude,
                             float_precision)

        f.clear()
        assert len(f.peaks) == 0

    def test_partials(self):
        N = 256
        f = base.Frame(N)
        f.max_partials = 10

        p = base.Peak()
        p.amplitude = 0.5
        p.frequency = 220.0
        p.phase = 0.0

        f.partial(0, p)
        assert_almost_equals(f.partial(0).amplitude, p.amplitude,
                             float_precision)
        assert_almost_equals(f.partial(0).frequency, p.frequency,
                             float_precision)
