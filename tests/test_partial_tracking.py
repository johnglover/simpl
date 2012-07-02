import os
import json
import numpy as np
from nose.tools import assert_almost_equals
import simpl
import simpl.peak_detection as peak_detection
import simpl.partial_tracking as partial_tracking

PeakDetection = peak_detection.PeakDetection
SMSPeakDetection = peak_detection.SMSPeakDetection
PartialTracking = partial_tracking.PartialTracking
SMSPartialTracking = partial_tracking.SMSPartialTracking

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
libsms_test_data_path = os.path.join(
    os.path.dirname(__file__), 'libsms_test_data.json'
)


def _load_libsms_test_data():
    test_data = None
    with open(libsms_test_data_path, 'r') as f:
        test_data = json.loads(f.read())
    return test_data


class TestPartialTracking(object):
    @classmethod
    def setup_class(cls):
        cls.audio = simpl.read_wav(audio_path)[0]
        cls.audio = cls.audio[0:num_samples]

    def test_basic(self):
        pd = PeakDetection()
        pd.hop_size = hop_size
        frames = pd.find_peaks(self.audio)

        pt = PartialTracking()
        frames = pt.find_partials(frames)

        print 'frames: %d (expected: %d)' %\
            (len(frames), len(self.audio) / hop_size)
        assert len(frames) == len(self.audio) / hop_size

        assert len(frames[0].partials) == 0
        assert frames[0].max_partials == 100


class TestSMSPartialTracking(object):
    @classmethod
    def setup_class(cls):
        cls.audio = simpl.read_wav(audio_path)[0]
        cls.audio = cls.audio[0:num_samples]
        cls.test_data = _load_libsms_test_data()

    def test_basic(self):
        pd = SMSPeakDetection()
        pd.hop_size = hop_size
        pd.max_peaks = max_peaks
        pd.static_frame_size = True
        frames = pd.find_peaks(self.audio)

        pt = SMSPartialTracking()
        pt.max_partials = max_partials
        frames = pt.find_partials(frames)

        print 'frames: %d (expected: %d)' %\
            (len(frames), len(self.audio) / hop_size)
        assert len(frames) == len(self.audio) / hop_size

        assert len(frames[0].partials) == 0
        assert frames[0].max_partials == max_partials

    def test_partial_tracking(self):
        pd = SMSPeakDetection()
        pd.max_peaks = max_peaks
        pd.hop_size = hop_size
        peaks = pd.find_peaks(self.audio)
        pt = SMSPartialTracking()
        pt.max_partials = max_partials
        frames = pt.find_partials(peaks)

        # make sure each partial is the same
        # for i in range(len(sms_frames)):
        #     assert len(sms_frames[i].partials) == len(simpl_frames[i].partials)
        #     for p in range(len(sms_frames[i].partials)):
        #         assert_almost_equals(sms_frames[i].partials[p].amplitude,
        #                              simpl_frames[i].partials[p].amplitude,
        #                              float_precision)
        #         assert_almost_equals(sms_frames[i].partials[p].frequency,
        #                              simpl_frames[i].partials[p].frequency,
        #                              float_precision)
        #         assert_almost_equals(sms_frames[i].partials[p].phase,
        #                              simpl_frames[i].partials[p].phase,
        #                              float_precision)
