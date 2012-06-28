import simpl
import matplotlib.pyplot as plt
import colours


def plot_frame_peaks(peaks):
    "Plot peaks in one frame"
    x_values = []
    y_values = []
    for peak in peaks:
        x_values.append(int(peak.frequency))
        y_values.append(peak.amplitude)
    plt.plot(x_values, y_values, 'ro')


def _plot_frame_peaks(peaks, frame_number, max_amp=0):
    "Plot one frame, which is a list of Peak objects"
    for peak in peaks:
        plt.plot(frame_number, int(peak.frequency), linestyle="None",
                 marker="o", markersize=2, markeredgewidth=None,
                 markerfacecolor=colours.pbj(peak.amplitude / max_amp))


def plot_peaks(frames):
    "Plot peaks found by a peak detection algorithm"
    # Get the maximum peak amplitude, used to select an appropriate
    # colour for each peak.
    max_amp = None
    for frame in frames:
        if frame.peaks:
            max_amp = max(max_amp, max([p.amplitude for p in frame.peaks]))
    # If no max amp then no peaks so return
    if not max_amp:
        print "Warning: no peaks with an amplitude of > 0 to plot, returning"
        return

    for frame_number, frame in enumerate(frames):
        _plot_frame_peaks(frame.peaks, frame_number, max_amp)


def plot_partials(frames, show_peaks=False):
    "Plot partials created by a partial tracking algorithm"
    # Get the maximum peak amplitude, used to select an appropriate
    # colour for each peak.
    max_amp = None
    for frame in frames:
        if frame.peaks:
            max_amp = max(max_amp, max([p.amplitude for p in frame.peaks]))
    # If no max amp then no peaks so return
    if not max_amp:
        print "Warning: no peaks with an amplitude of > 0 to plot, returning"
        return

    # Create Partial objects from frames
    num_frames = len(frames)
    partials = []
    live_partials = [None for i in range(len(frames[0].partials))]
    for n, f in enumerate(frames):
        for i, p in enumerate(f.partials):
            if p.amplitude > 0:
                # active partial
                if live_partials[i]:
                    live_partials[i].add_peak(p)
                else:
                    partial = simpl.Partial()
                    partial.starting_frame = n
                    partial.add_peak(p)
                    live_partials[i] = partial
            else:
                # currently dead
                if live_partials[i]:
                    partials.append(live_partials[i])
                    live_partials[i] = None
    for p in live_partials:
        if p:
            partials.append(p)

    peaks = [[] for f in range(num_frames)]
    for partial in partials:
        x_values = []
        y_values = []
        avg_amp = 0.0
        num_peaks = 0
        for peak_number, peak in enumerate(partial.peaks):
            x_values.append(partial.starting_frame + peak_number)
            y_values.append(int(peak.frequency))
            avg_amp += peak.amplitude
            num_peaks += 1
            peaks[partial.starting_frame + peak_number].append(peak)
        avg_amp /= num_peaks
        plt.plot(x_values, y_values, color=colours.pbj(avg_amp / max_amp))

    if show_peaks:
        plot_peaks(frames)
