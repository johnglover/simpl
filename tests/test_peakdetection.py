import simpl
import numpy as np
from scipy.io.wavfile import read


class TestPeakDetection(object):
    frame_size = 2048
    hop_size = 512
    max_peaks = 10
