import simpl
import loris
import scipy as sp
from scipy.io.wavfile import read
from nose.tools import assert_almost_equals

class TestLoris(object):
    def setUp(self):
        input_file = 'audio/flute.wav'
        audio_data = read(input_file)
        self.audio = simpl.asarray(audio_data[1]) / 32768.0
        self.sample_rate = audio_data[0]
        self.FUNDAMENTAL = 415.0 	

    def test_peak_detection(self):
        pass

    def test_partial_tracking(self):
        pass

    def test_full_analysis(self):
        # configure a Loris Analyzer, use frequency
        # resolution equal to 80% of the fundamental
        # frequency, main lobe width equal to the 
        # fundamental frequency, and frequency drift 
        # equal to 20% of the fundamental frequency
        # (approx. 83 Hz )
        loris_analyzer = loris.Analyzer(.8 * self.FUNDAMENTAL, self.FUNDAMENTAL)
        loris_analyzer.setFreqDrift(.2 * self.FUNDAMENTAL)
        loris_partials = loris_analyzer.analyze(self.audio, self.sample_rate)
        simpl_analyzer = simpl.loris.Analyzer(.8 * self.FUNDAMENTAL, self.FUNDAMENTAL)
        simpl_analyzer.setFreqDrift(.2 * self.FUNDAMENTAL)
        simpl_partials = simpl_analyzer.analyze(self.audio, self.sample_rate)

if __name__ == "__main__":
    input_file = 'audio/flute.wav'
    FUNDAMENTAL = 415.0 	
    audio_data = read(input_file)
    audio = simpl.asarray(audio_data[1]) / 32768.0
    sample_rate = audio_data[0]

    an = simpl.loris.Analyzer(.8 * FUNDAMENTAL, FUNDAMENTAL)
    an.setFreqDrift(.2 * FUNDAMENTAL)

    # analyze and store partials
    peaks = an.analyze_peaks(audio, sample_rate)
