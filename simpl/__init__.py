import numpy as np
dtype = np.double

import base
Frame = base.Frame
Peak = base.Peak

import peak_detection
PeakDetection = peak_detection.PeakDetection
SMSPeakDetection = peak_detection.SMSPeakDetection

import partial_tracking
PartialTracking = partial_tracking.PartialTracking
SMSPartialTracking = partial_tracking.SMSPartialTracking

import pybase
Partial = pybase.Partial
# PartialTracking = pybase.PartialTracking
Synthesis = pybase.Synthesis
Residual = pybase.Residual
compare_peak_amps = pybase.compare_peak_amps
compare_peak_freqs = pybase.compare_peak_freqs

import sndobj
SndObjPeakDetection = sndobj.SndObjPeakDetection
SndObjPartialTracking = sndobj.SndObjPartialTracking
SndObjSynthesis = sndobj.SndObjSynthesis

import pysms
# SMSPartialTracking = pysms.SMSPartialTracking
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
