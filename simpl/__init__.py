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

import synthesis
Synthesis = synthesis.Synthesis
SMSSynthesis = synthesis.SMSSynthesis

import residual
Residual = residual.Residual
SMSResidual = residual.SMSResidual

import pybase
Partial = pybase.Partial
Residual = pybase.Residual
compare_peak_amps = pybase.compare_peak_amps
compare_peak_freqs = pybase.compare_peak_freqs

import pysndobj
SndObjPeakDetection = pysndobj.SndObjPeakDetection
SndObjPartialTracking = pysndobj.SndObjPartialTracking
SndObjSynthesis = pysndobj.SndObjSynthesis

import mq
MQPeakDetection = mq.MQPeakDetection
MQPartialTracking = mq.MQPartialTracking
MQSynthesis = mq.MQSynthesis

import plot
plot_peaks = plot.plot_peaks
plot_partials = plot.plot_partials

import audio
read_wav = audio.read_wav
