import numpy as np
import base
import peak_detection
import partial_tracking
import synthesis
import residual
import plot
import audio
import pybase

dtype = np.double
Frame = base.Frame
Peak = base.Peak
PeakDetection = peak_detection.PeakDetection
SMSPeakDetection = peak_detection.SMSPeakDetection
SndObjPeakDetection = peak_detection.SndObjPeakDetection
PartialTracking = partial_tracking.PartialTracking
SMSPartialTracking = partial_tracking.SMSPartialTracking
SndObjPartialTracking = partial_tracking.SndObjPartialTracking
Synthesis = synthesis.Synthesis
SMSSynthesis = synthesis.SMSSynthesis
Residual = residual.Residual
SMSResidual = residual.SMSResidual
plot_peaks = plot.plot_peaks
plot_partials = plot.plot_partials
read_wav = audio.read_wav
Partial = pybase.Partial
compare_peak_amps = pybase.compare_peak_amps
compare_peak_freqs = pybase.compare_peak_freqs

import pysndobj
SndObjSynthesis = pysndobj.SndObjSynthesis

import mq
MQPeakDetection = mq.MQPeakDetection
MQPartialTracking = mq.MQPartialTracking
MQSynthesis = mq.MQSynthesis
