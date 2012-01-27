from basetypes import Frame, Peak, Partial
from basetypes import PeakDetection, PartialTracking, Synthesis, Residual
from basetypes import compare_peak_amps, compare_peak_freqs
from sndobj import SndObjPeakDetection, SndObjPartialTracking, SndObjSynthesis
from sms import SMSPeakDetection, SMSPartialTracking, SMSSynthesis, SMSResidual
from loris import LorisPeakDetection, LorisPartialTracking, LorisSynthesis
from mq import MQPeakDetection, MQPartialTracking, MQSynthesis
from plot import plot_peaks, plot_partials
from audio import read_wav

import numpy as np

def array (n, type=float):
    return(np.array(n, dtype=type))

def asarray (n, type=float):
    return(np.asarray(n, dtype=type))

def zeros (n, type=float):
    return(np.zeros(n, dtype=type))
