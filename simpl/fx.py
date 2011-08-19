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

from simpl import Partial
import numpy as np

def time_stretch(partials, factor):
    """Time stretch partials by factor."""
    stretched_partials = []
    step_size = 1.0 / factor
            
    for partial in partials:
        stretched_partial = Partial()
        stretched_partial.starting_frame = partial.starting_frame * factor
        stretched_partial.partial_id = partial.partial_id
        num_steps = int((partial.get_length() - 1) / step_size)
        current_step = 0
        for step in range(num_steps):
            current_peak = partial.peaks[int(np.floor(current_step))]
            stretched_partial.add_peak(current_peak)
            current_step += step_size
        stretched_partials.append(stretched_partial)
    return stretched_partials
