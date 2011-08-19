# Copyright (c) 2009-2011 John Glover, National University of Ireland, Maynooth
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

import numpy as np
import simpl
from simpl.simplloris import LorisPeakDetection

# class LorisPeakDetection(simpl.PeakDetection):
#     "Sinusoidal peak detection using Loris"
#     def __init__(self):
#         simpl.PeakDetection.__init__(self)
        

class LorisPartialTracking(simpl.PartialTracking):
    "Partial tracking using the algorithm from Loris"
    def __init__(self):
        simpl.PartialTracking.__init__(self)
        
        
class LorisSynthesis(simpl.Synthesis):
    "Sinusoidal resynthesis using Loris"
    def __init__(self, synthesis_type='adsyn'):
        simpl.Synthesis.__init__(self)
