import matplotlib.pyplot as plt
import simpl.plot.colours


def plot_peaks(frames):
    "Plot peaks found by a peak detection algorithm"
    # Get the maximum peak amplitude, used to select an appropriate
    # colour for each peak.
    max_amp = 0
    for frame in frames:
        if frame.peaks:
            max_amp = max(max_amp, max([p.amplitude for p in frame.peaks]))

    if not max_amp:
        print("No peaks with an amplitude of > 0 to plot")
        return

    for frame_number, frame in enumerate(frames):
        for peak in frame.peaks:
            plt.plot(frame_number, int(peak.frequency), linestyle="None",
                     marker="o", markersize=2, markeredgewidth=None,
                     markerfacecolor=colours.pbj(peak.amplitude / max_amp))


def plot_partials(frames, show_peaks=False):
    "Plot partials created by a partial tracking algorithm"
    # Get the maximum peak amplitude, used to select an appropriate
    # colour for each peak.
    max_amp = 0
    for frame in frames:
        if frame.partials:
            max_amp = max(max_amp, max([p.amplitude for p in frame.partials]))

    if not max_amp:
        print("No partial peaks with an amplitude of > 0 to plot")
        return

    for n in range(len(frames) - 1):
        for p in range(len(frames[n].partials)):
            x = [n, n + 1]
            y = [frames[n].partials[p].frequency,
                 frames[n + 1].partials[p].frequency]
            amp = frames[n].partials[p].amplitude
            freq = frames[n + 1].partials[p].frequency
            if amp and freq:
                plt.plot(x, y, color=colours.pbj(amp / max_amp))

    if show_peaks:
        plot_peaks(frames)
