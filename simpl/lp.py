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

import simpl
import numpy as np

def burg(signal, order):
    """Using the burg method, calculate order coefficients.
    Returns a numpy array."""
    coefs = np.array([1.0])
    # initialise f and b - the forward and backwards predictors
    f = np.zeros(len(signal))
    b = np.zeros(len(signal))
    for i in range(len(signal)):
        f[i] = b[i] = signal[i]
    # burg algorithm
    for k in range(order):
        # fk is f without the first element
        fk = f[1:]
        # bk is b without the last element
        bk = b[0:b.size-1]
        # calculate mu
        if sum((fk*fk)+(bk*bk)):
            # check for division by zero
            mu = -2.0 * sum(fk*bk) / sum((fk*fk)+(bk*bk))
        else:
            mu = 0.0
        # update coefs
        # coefs[::-1] just reverses the coefs array
        coefs = np.hstack((coefs,0)) + (mu * np.hstack((0, coefs[::-1])))
        # update f and b
        f = fk + (mu*bk)
        b = bk + (mu*fk)
    return coefs[1:]

def predict(signal, coefs, num_predictions):
    """Using Linear Prediction, return the estimated next num_predictions
    values of signal, using the given coefficients.
    Returns a numpy array."""
    predictions = np.zeros(num_predictions)
    past_samples = np.zeros(len(coefs))
    for i in range(len(coefs)):
        past_samples[i] = signal[-1-i]
    sample_pos = 0
    for i in range(num_predictions):
        # each sample in past_samples is multiplied by a coefficient
        # results are summed
        for j in range(len(coefs)):
            predictions[i] -= coefs[j] * past_samples[(j+sample_pos) % len(coefs)]
        sample_pos -= 1
        if sample_pos < 0:
            sample_pos = len(coefs) - 1
        past_samples[sample_pos] = predictions[i] 
    return predictions


class LPPartialTracking(simpl.PartialTracking):
    "Partial tracking, based on the McAulay and Quatieri (MQ) algorithm"    
    def __init__(self):
        simpl.PartialTracking.__init__(self)
        self._matching_interval = 100  # peak matching interval, in Hz

        
    def update_partials(self, frame, frame_number):
        """Streamable (real-time) LP peak-tracking.
        """
        frame_partials = []
                
        for peak in frame:
            pass

        return frame_partials

