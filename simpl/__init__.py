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
Partial = pybase.Partial
compare_peak_amps = pybase.compare_peak_amps
compare_peak_freqs = pybase.compare_peak_freqs
read_wav = audio.read_wav

PeakDetection = peak_detection.PeakDetection
SMSPeakDetection = peak_detection.SMSPeakDetection
SndObjPeakDetection = peak_detection.SndObjPeakDetection
LorisPeakDetection = peak_detection.LorisPeakDetection

PartialTracking = partial_tracking.PartialTracking
SMSPartialTracking = partial_tracking.SMSPartialTracking
SndObjPartialTracking = partial_tracking.SndObjPartialTracking
LorisPartialTracking = partial_tracking.LorisPartialTracking

Synthesis = synthesis.Synthesis
SMSSynthesis = synthesis.SMSSynthesis
SndObjSynthesis = synthesis.SndObjSynthesis
LorisSynthesis = synthesis.LorisSynthesis

Residual = residual.Residual
SMSResidual = residual.SMSResidual

plot_peaks = plot.plot_peaks
plot_partials = plot.plot_partials

import mq
MQPeakDetection = mq.MQPeakDetection
MQPartialTracking = mq.MQPartialTracking
MQSynthesis = mq.MQSynthesis
