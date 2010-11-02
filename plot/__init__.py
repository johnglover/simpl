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

#def _plot_frame_peaks(frame, frame_number, max_amp=0):
#    "Plot one frame, which is a list of Peak objects"
#    x_values = [frame_number for x in range(len(frame))]
#    y_values = [int(peak.frequency) for peak in frame]
#    plt.plot(x_values, y_values, 'ro')

def _plot_frame_peaks(frame, frame_number, max_amp=0):
    "Plot one frame, which is a list of Peak objects"
    for peak in frame:
        plt.plot(frame_number, int(peak.frequency), linestyle="None",
                 marker="o", markersize=2, markeredgewidth=None,
                 markerfacecolor=colours.pbj(peak.amplitude/max_amp))
    
def plot_peaks(peaks):
    "Plot peaks found by a peak detection algorithm"
    max_amp = max([max([p.amplitude for p in f if p]) for f in peaks if f])
    for frame_number, frame in enumerate(peaks):
        _plot_frame_peaks(frame, frame_number, max_amp)
        
#def plot_partials(partials, show_peaks=True):
#    "Plot partials created by a partial tracking algorithm"
#    max_amp = max([max([peak.amplitude for peak in p.peaks if p.peaks])
#                   for p in partials if p])
#    num_frames = max([partial.get_last_frame() for partial in partials])

#    peaks = [[] for f in range(num_frames)]
#    for partial in partials:
#        x_values = []
#        y_values = []
#        for peak_number, peak in enumerate(partial.peaks):
#            x_values.append(partial.starting_frame + peak_number)
#            y_values.append(int(peak.frequency))
#            peaks[partial.starting_frame + peak_number].append(peak)
#        plt.plot(x_values, y_values, "b")
#    if show_peaks:
#        plot_peaks(peaks)  

def plot_partials(partials, show_peaks=False):
    "Plot partials created by a partial tracking algorithm"
    max_amp = max([max([peak.amplitude for peak in p.peaks if p.peaks])
                   for p in partials if p])
    num_frames = max([partial.get_last_frame() for partial in partials])

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
        plt.plot(x_values, y_values, color=colours.pbj(avg_amp/max_amp))
    if show_peaks:
        plot_peaks(peaks)  

