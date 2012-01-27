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
