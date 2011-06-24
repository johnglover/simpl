# Copyright (c) 2010 John Glover, National University of Ireland, Maynooth
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

def print_peaks(frames):
    for n, f in enumerate(frames):
        for p in f:
            print str(n) + ":", p.frequency

def print_partials(partials):
    for partial_num, partial in enumerate(partials):
        print partial_num,
        print "(" + str(partial.starting_frame) + " to",
        print str(partial.starting_frame + len(partial.peaks)) + "):",
        for peak_number, peak in enumerate(partial.peaks):
            print peak.frequency,
        print
