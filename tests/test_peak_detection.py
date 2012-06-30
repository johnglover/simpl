import os
import numpy as np
from nose.tools import assert_almost_equals
import pysms
import simpl
import simpl.peak_detection as peak_detection

PeakDetection = peak_detection.PeakDetection
SMSPeakDetection = peak_detection.SMSPeakDetection

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


class TestPeakDetection(object):
    @classmethod
    def setup_class(cls):
        cls.audio = simpl.read_wav(audio_path)[0]

    def test_peak_detection(self):
        pd = PeakDetection()
        pd.find_peaks(self.audio)

        assert len(pd.frames) == len(self.audio) / hop_size
        assert len(pd.frames[0].peaks) == 0


class TestSMSPeakDetection(object):
    def _pysms_analysis_params(self, sampling_rate):
        analysis_params = pysms.SMS_AnalParams()
        pysms.sms_initAnalParams(analysis_params)
        analysis_params.iSamplingRate = sampling_rate
        analysis_params.iFrameRate = sampling_rate / hop_size
        analysis_params.iWindowType = pysms.SMS_WIN_HAMMING
        analysis_params.fDefaultFundamental = 100
        analysis_params.fHighestFreq = 20000
        analysis_params.iFormat = pysms.SMS_FORMAT_HP
        analysis_params.nTracks = max_peaks
        analysis_params.peakParams.iMaxPeaks = max_peaks
        analysis_params.nGuides = max_peaks
        analysis_params.iMaxDelayFrames = 4
        analysis_params.analDelay = 0
        analysis_params.minGoodFrames = 1
        analysis_params.iCleanTracks = 0
        analysis_params.iStochasticType = pysms.SMS_STOC_NONE
        analysis_params.preEmphasis = 0
        return analysis_params

    def test_size_next_read(self):
        """
        test_size_next_read
        Make sure PeakDetection is calculating the correct value for the
        size of the next frame.
        """
        audio, sampling_rate = simpl.read_wav(audio_path)
        pysms.sms_init()
        snd_header = pysms.SMS_SndHeader()

        # Try to open the input file to fill snd_header
        if(pysms.sms_openSF(audio_path, snd_header)):
            raise NameError(
                "error opening sound file: " + pysms.sms_errorString()
            )

        analysis_params = self._pysms_analysis_params(sampling_rate)
        analysis_params.iMaxDelayFrames = num_frames + 1
        if pysms.sms_initAnalysis(analysis_params, snd_header) != 0:
            raise Exception("Error allocating memory for analysis_params")
        analysis_params.nFrames = num_frames
        sms_header = pysms.SMS_Header()
        pysms.sms_fillHeader(sms_header, analysis_params, "pysms")

        sample_offset = 0
        pysms_size_new_data = 0
        current_frame = 0
        sms_next_read_sizes = []

        while current_frame < num_frames:
            sms_next_read_sizes.append(analysis_params.sizeNextRead)
            sample_offset += pysms_size_new_data
            pysms_size_new_data = analysis_params.sizeNextRead

            # convert frame to floats for libsms
            frame = audio[sample_offset:sample_offset + pysms_size_new_data]
            frame = np.array(frame, dtype=np.float32)
            if len(frame) < pysms_size_new_data:
                frame = np.hstack((
                    frame, np.zeros(pysms_size_new_data - len(frame),
                                    dtype=np.float32)
                ))

            analysis_data = pysms.SMS_Data()
            pysms.sms_allocFrameH(sms_header, analysis_data)
            status = pysms.sms_analyze(frame, analysis_data, analysis_params)
            # as the no. of frames of delay is > num_frames, sms_analyze should
            # never get around to performing partial tracking, and so the
            # return value should be 0
            assert status == 0
            pysms.sms_freeFrame(analysis_data)
            current_frame += 1

        pysms.sms_freeAnalysis(analysis_params)
        pysms.sms_closeSF()
        pysms.sms_free()

        pd = SMSPeakDetection()
        pd.hop_size = hop_size
        pd.max_peaks = max_peaks
        current_frame = 0
        sample_offset = 0

        while current_frame < num_frames:
            pd.frame_size = pd.next_frame_size()
            assert sms_next_read_sizes[current_frame] == pd.frame_size,\
                (sms_next_read_sizes[current_frame], pd.frame_size)
            frame = simpl.Frame()
            frame.size = pd.frame_size
            frame.audio = audio[sample_offset:sample_offset + pd.frame_size]
            pd.find_peaks_in_frame(frame)
            sample_offset += pd.frame_size
            current_frame += 1
