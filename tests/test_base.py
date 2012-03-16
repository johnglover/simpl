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
