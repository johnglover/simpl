import numpy as np
dtype = np.double

import pybase
Frame = pybase.Frame
Peak = pybase.Peak
Partial = pybase.Partial
PeakDetection = pybase.PeakDetection
PartialTracking = pybase.PartialTracking
Synthesis = pybase.Synthesis
Residual = pybase.Residual
compare_peak_amps = pybase.compare_peak_amps
compare_peak_freqs = pybase.compare_peak_freqs

import sndobj
SndObjPeakDetection = sndobj.SndObjPeakDetection
SndObjPartialTracking = sndobj.SndObjPartialTracking
SndObjSynthesis = sndobj.SndObjSynthesis

import pysms
SMSPeakDetection = pysms.SMSPeakDetection
SMSPartialTracking = pysms.SMSPartialTracking
SMSSynthesis = pysms.SMSSynthesis
SMSResidual = pysms.SMSResidual

import mq
MQPeakDetection = mq.MQPeakDetection
MQPartialTracking = mq.MQPartialTracking
MQSynthesis = mq.MQSynthesis

import plot
plot_peaks = plot.plot_peaks
plot_partials = plot.plot_partials

import audio
read_wav = audio.read_wav
