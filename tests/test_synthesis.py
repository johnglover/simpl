import os
from nose.tools import assert_almost_equals
import simpl
import simpl.peak_detection as peak_detection
import simpl.partial_tracking as partial_tracking
import simpl.synthesis as synthesis

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
libsms_harmonic_synthesis_ifft_path = os.path.join(
    os.path.dirname(__file__), 'libsms_harmonic_synthesis_ifft.wav'
)
libsms_harmonic_synthesis_sin_path = os.path.join(
    os.path.dirname(__file__), 'libsms_harmonic_synthesis_sin.wav'
)

PeakDetection = peak_detection.PeakDetection
SMSPeakDetection = peak_detection.SMSPeakDetection
PartialTracking = partial_tracking.PartialTracking
SMSPartialTracking = partial_tracking.SMSPartialTracking
Synthesis = synthesis.Synthesis
SMSSynthesis = synthesis.SMSSynthesis


class TestSynthesis(object):
    @classmethod
    def setup_class(cls):
        cls.audio = simpl.read_wav(audio_path)[0]
        cls.audio = cls.audio[0:num_samples]

    def test_basic(self):
        pd = PeakDetection()
        pd.hop_size = hop_size
        frames = pd.find_peaks(self.audio)

        pt = PartialTracking()
        pt.max_partials = max_partials
        frames = pt.find_partials(frames)

        s = Synthesis()
        s.hop_size = hop_size
        synth_audio = s.synth(frames)

        assert len(synth_audio) == len(self.audio),\
            (len(synth_audio), len(self.audio))


class TestSMSSynthesis(object):
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

        s = SMSSynthesis()
        s.hop_size = hop_size
        synth_audio = s.synth(frames)

        assert len(synth_audio) == len(self.audio),\
            (len(synth_audio), len(self.audio))

    def test_harmonic_synthesis_ifft(self):
        pd = SMSPeakDetection()
        pd.hop_size = hop_size
        frames = pd.find_peaks(self.audio)

        pt = SMSPartialTracking()
        pt.max_partials = max_partials
        frames = pt.find_partials(frames)

        synth = SMSSynthesis()
        synth.hop_size = hop_size
        synth.max_partials = max_partials
        synth.det_synthesis_type = SMSSynthesis.SMS_DET_IFFT
        synth_audio = synth.synth(frames)

        assert len(synth_audio) == len(self.audio)

        sms_audio, sampling_rate = simpl.read_wav(
            libsms_harmonic_synthesis_ifft_path
        )

        assert len(synth_audio) == len(sms_audio)

        for i in range(len(synth_audio)):
            assert_almost_equals(synth_audio[i], sms_audio[i], float_precision)

    def test_harmonic_synthesis_sin(self):
        pd = SMSPeakDetection()
        pd.hop_size = hop_size
        frames = pd.find_peaks(self.audio)

        pt = SMSPartialTracking()
        pt.max_partials = max_partials
        frames = pt.find_partials(frames)

        synth = SMSSynthesis()
        synth.hop_size = hop_size
        synth.max_partials = max_partials
        synth.det_synthesis_type = SMSSynthesis.SMS_DET_SIN
        synth_audio = synth.synth(frames)

        assert len(synth_audio) == len(self.audio)

        sms_audio, sampling_rate = simpl.read_wav(
            libsms_harmonic_synthesis_sin_path
        )

        assert len(synth_audio) == len(sms_audio)

        for i in range(len(synth_audio)):
            assert_almost_equals(synth_audio[i], sms_audio[i], float_precision)
