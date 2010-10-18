# Copyright (c) 2009 John Glover, National University of Ireland, Maynooth
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

from basetypes import Peak, PeakDetection, Partial, PartialTracking, Synthesis, Residual
from basetypes import compare_peak_amps, compare_peak_freqs
from sndobj import SndObjPeakDetection, SndObjPartialTracking, SndObjSynthesis
from sms import SMSPeakDetection, SMSPartialTracking, SMSSynthesis, SMSResidual
from mq import MQPeakDetection, MQPartialTracking

from numpy import array as np_array
def array (n, type='float32'):
    return(np_array(n, type))

from numpy import zeros as np_zeros
def zeros (n, type='float32'):
    return(np_zeros(n, type))
