import simpl.base as base
import numpy as np


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

        f.clear_peaks()
        assert f.num_peaks == 0


class TestPeakDetection(object):
    def test_peak_detection(self):
        pd = base.PeakDetection()
        print pd
