import os
import json
import simpl
import simpl.peak_detection as peak_detection

PeakDetection = peak_detection.PeakDetection
SMSPeakDetection = peak_detection.SMSPeakDetection
SndObjPeakDetection = peak_detection.SMSPeakDetection

float_precision = 5
frame_size = 512
hop_size = 512
max_peaks = 10
max_partials = 10
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

    def test_basic(self):
        pd = PeakDetection()
        pd.max_peaks = max_peaks
        pd.find_peaks(self.audio)

        assert len(pd.frames) == len(self.audio) / hop_size
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
        pd.static_frame_size = True
        pd.find_peaks(self.audio)

        assert len(pd.frames) == len(self.audio) / hop_size
        assert len(pd.frames[0].peaks)

    def test_size_next_read(self):
        audio, sampling_rate = simpl.read_wav(audio_path)

        pd = SMSPeakDetection()
        pd.hop_size = hop_size
        pd.static_frame_size = False
        pd.max_peaks = max_peaks
        current_frame = 0
        sample_offset = 0

        next_read_sizes = self.test_data['size_next_read']

        while current_frame < num_frames:
            pd.frame_size = pd.next_frame_size()
            assert next_read_sizes[current_frame] == pd.frame_size,\
                (next_read_sizes[current_frame], pd.frame_size)
            frame = simpl.Frame()
            frame.size = pd.frame_size
            frame.audio = audio[sample_offset:sample_offset + pd.frame_size]
            pd.find_peaks_in_frame(frame)
            sample_offset += pd.frame_size
            current_frame += 1

    def test_peak_detection(self):
        audio, sampling_rate = simpl.read_wav(audio_path)

        pd = SMSPeakDetection()
        pd.max_peaks = max_peaks
        pd.hop_size = hop_size
        frames = pd.find_peaks(audio[0:num_samples])

        sms_frames = self.test_data['peak_detection']
        sms_frames = [f for f in sms_frames if f['status'] != 0]

        print 'frames: %d (expected: %d)' % (len(frames), len(sms_frames))
        assert len(sms_frames) == len(frames)

        for frame in frames:
            assert frame.num_peaks <= max_peaks, frame.num_peaks
            max_amp = max([p.amplitude for p in frame.peaks])
            assert max_amp


class TestSndObjPeakDetection(object):
    def test_peak_detection(self):
        audio, sampling_rate = simpl.read_wav(audio_path)

        pd = SndObjPeakDetection()
        pd.max_peaks = max_peaks
        pd.hop_size = hop_size
        frames = pd.find_peaks(audio[0:num_samples])

        assert len(frames) == num_samples / hop_size

        for frame in frames:
            assert frame.num_peaks <= max_peaks, frame.num_peaks
            max_amp = max([p.amplitude for p in frame.peaks])
            assert max_amp
