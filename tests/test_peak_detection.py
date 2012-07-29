import os
import json
import simpl
import simpl.peak_detection as peak_detection

PeakDetection = peak_detection.PeakDetection
SMSPeakDetection = peak_detection.SMSPeakDetection
SndObjPeakDetection = peak_detection.SndObjPeakDetection

float_precision = 5
hop_size = 512
max_peaks = 10
num_frames = 30
num_samples = num_frames * hop_size
audio_path = os.path.join(
    os.path.dirname(__file__), 'audio/flute.wav'
)
test_data_path = os.path.join(
    os.path.dirname(__file__), 'libsms_test_data.json'
)


def _load_libsms_test_data():
    test_data = None
    with open(test_data_path, 'r') as f:
        test_data = json.loads(f.read())
    return test_data


class TestPeakDetection(object):
    @classmethod
    def setup_class(cls):
        cls.audio = simpl.read_wav(audio_path)[0]
        cls.audio = cls.audio[0:num_samples]

    def test_basic(self):
        pd = PeakDetection()
        pd.max_peaks = max_peaks
        pd.find_peaks(self.audio)

        assert len(pd.frames) == num_frames
        assert len(pd.frames[0].peaks) == 0
        assert pd.frames[0].max_peaks == max_peaks


class TestSMSPeakDetection(object):
    @classmethod
    def setup_class(cls):
        cls.audio = simpl.read_wav(audio_path)[0]
        cls.test_data = _load_libsms_test_data()

    def test_basic(self):
        pd = SMSPeakDetection()
        pd.hop_size = hop_size
        pd.frame_size = hop_size
        pd.static_frame_size = True
        pd.find_peaks(self.audio)

        assert len(pd.frames) == \
            ((len(self.audio) - pd.frame_size) / hop_size) + 1
        assert len(pd.frames[0].peaks)

    def test_size_next_read(self):
        audio, sampling_rate = simpl.read_wav(audio_path)

        pd = SMSPeakDetection()
        pd.hop_size = hop_size
        pd.max_peaks = max_peaks

        sizes = self.test_data['size_next_read']
        frames = pd.find_peaks(audio[0:num_samples])

        for i, frame in enumerate(frames):
            assert sizes[i] == frame.size, (sizes[i], frame.size)

    def test_peak_detection(self):
        audio, sampling_rate = simpl.read_wav(audio_path)

        pd = SMSPeakDetection()
        pd.hop_size = hop_size
        pd.max_peaks = max_peaks
        frames = pd.find_peaks(audio[0:num_samples])

        sms_frames = self.test_data['peak_detection']
        sms_frames = [f for f in sms_frames if f['status'] != 0]

        # assert len(sms_frames) == len(frames)

        for frame in frames:
            assert len(frame.peaks) <= max_peaks, len(frame.peaks)
            max_amp = max([p.amplitude for p in frame.peaks])
            assert max_amp


class TestSndObjPeakDetection(object):
    @classmethod
    def setup_class(cls):
        cls.audio = simpl.read_wav(audio_path)[0]
        cls.audio = cls.audio[len(cls.audio) / 2:
                              (len(cls.audio) / 2) + num_samples]

    def test_peak_detection(self):
        pd = SndObjPeakDetection()
        pd.hop_size = hop_size
        pd.max_peaks = max_peaks
        frames = pd.find_peaks(self.audio)

        assert len(frames) == num_frames

        # last 4 frames could be empty (with default frame size of 2048
        # and hop size of 512) so ignore
        frames = frames[0:len(frames) - 4]

        for frame in frames:
            assert len(frame.peaks) <= max_peaks, len(frame.peaks)
            max_amp = max([p.amplitude for p in frame.peaks])
            assert max_amp
