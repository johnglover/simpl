import os
import numpy as np
from nose.tools import assert_almost_equals
import simpl
import simpl.peak_detection as peak_detection
import simpl.partial_tracking as partial_tracking
import simpl.synthesis as synthesis
import simpl.residual as residual

float_precision = 2
frame_size = 512
hop_size = 512
max_peaks = 10
max_partials = 10
num_frames = 30
num_samples = num_frames * hop_size
audio_path = os.path.join(
    os.path.dirname(__file__), 'audio/flute.wav'
)
libsms_residual_synthesis_path = os.path.join(
    os.path.dirname(__file__), 'libsms_residual_synthesis.wav'
)

PeakDetection = peak_detection.PeakDetection
SMSPeakDetection = peak_detection.SMSPeakDetection
PartialTracking = partial_tracking.PartialTracking
SMSPartialTracking = partial_tracking.SMSPartialTracking
Synthesis = synthesis.Synthesis
SMSSynthesis = synthesis.SMSSynthesis
Residual = residual.Residual
SMSResidual = residual.SMSResidual


class TestResidual(object):
    @classmethod
    def setup_class(cls):
        cls.audio = simpl.read_wav(audio_path)[0]

    def test_basic(self):
        pd = PeakDetection()
        frames = pd.find_peaks(self.audio)

        pt = PartialTracking()
        frames = pt.find_partials(frames)

        synth = Synthesis()
        synth_audio = synth.synth(frames)

        res = Residual()
        residual_audio = res.find_residual(synth_audio, self.audio)
        assert len(residual_audio)


class TestSMSResidual(object):
    @classmethod
    def setup_class(cls):
        cls.audio = simpl.read_wav(audio_path)[0]
        cls.audio = cls.audio[0:num_samples]

    def test_basic(self):
        pd = SMSPeakDetection()
        pd.hop_size = hop_size
        frames = pd.find_peaks(self.audio)

        pt = SMSPartialTracking()
        pt.max_partials = max_partials
        frames = pt.find_partials(frames)

        synth = SMSSynthesis()
        synth.hop_size = hop_size
        synth_audio = synth.synth(frames)

        res = SMSResidual()
        residual_audio = res.find_residual(synth_audio, self.audio)
        assert len(residual_audio)

    def test_residual_synthesis(self):
        res = SMSResidual()
        res.hop_size = hop_size
        simpl_residual = res.synth(self.audio)

        sms_residual, sampling_rate = simpl.read_wav(
            libsms_residual_synthesis_path
        )

        assert len(simpl_residual) == len(sms_residual)

        import matplotlib.pyplot as plt
        plt.plot(simpl_residual)
        plt.plot(sms_residual)
        plt.show()

        for i in range(len(simpl_residual)):
            assert_almost_equals(simpl_residual[i], sms_residual[i],
                                 float_precision)
