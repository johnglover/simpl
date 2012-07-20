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


def plot_partials(frames, show_peaks=True):
    "Plot partials created by a partial tracking algorithm"
    # Get the maximum peak amplitude, used to select an appropriate
    # colour for each peak.
    max_amp = None
    for frame in frames:
        if frame.partials:
            max_amp = max(max_amp, max([p.amplitude for p in frame.partials]))

    if not max_amp:
        print "No partial peaks with an amplitude of > 0 to plot"
        return

    for n in range(len(frames) - 1):
        for p in range(frame.max_partials):
            x = [n, n + 1]
            y = [frames[n].partial(p).frequency,
                 frames[n + 1].partial(p).frequency]
            amp = frames[n].partial(p).amplitude
            plt.plot(x, y, color=colours.pbj(amp / max_amp))

    if show_peaks:
        plot_peaks(frames)
