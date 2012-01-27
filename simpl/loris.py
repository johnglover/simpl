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
