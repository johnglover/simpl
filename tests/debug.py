
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

