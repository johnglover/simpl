#include "math.h"
#include "twm.h"

using namespace simpl;


int simpl::best_match(sample freq, std::vector<sample> candidates) {
    sample best_diff = 22050.0;
    sample diff = 0.0;
    int best = 0;

    for(int i = 0; i < candidates.size(); i++) {
        diff = fabs(freq - candidates[i]);
        if(diff < best_diff) {
            best_diff = diff;
            best = i;
        }
    }

    return best;
}

sample simpl::twm(Peaks peaks, sample f_min, sample f_max, sample f_step) {
    sample p = 0.5;
    sample q = 1.4;
    sample r = 0.5;
    sample rho = 0.33;
    int N = 30;
    std::map<sample, sample> err;

    if(peaks.size() == 0) {
        return 0.0;
    }

    sample max_amp = 0.0;
    for(int i = 0; i < peaks.size(); i++) {
        if(peaks[i]->amplitude > max_amp) {
            max_amp = peaks[i]->amplitude;
        }
    }

    if(max_amp == 0) {
        return 0.0;
    }

    // remove all peaks with amplitude of less than 10% of max
    // note: this is not in the TWM paper, found that it improved
    // accuracy however
    for(int i = 0; i < peaks.size(); i++) {
        if(peaks[i]->amplitude < (max_amp * 0.1)) {
            peaks.erase(peaks.begin() + i);
        }
    }

    // get the max frequency of the remaining peaks
    sample max_freq = 0.0;
    for(int i = 0; i < peaks.size(); i++) {
        if(peaks[i]->frequency > max_freq) {
            max_freq = peaks[i]->frequency;
        }
    }

    std::vector<sample> peak_freqs;
    for(int i = 0; i < peaks.size(); i++) {
        peak_freqs.push_back(peaks[i]->frequency);
    }

    sample f_current = f_min;
    while(f_current < f_max) {
        sample err_pm = 0.0;
        sample err_mp = 0.0;
        std::vector<sample> harmonics;

        for(sample f = f_current; f <= f_max; f += f_current) {
            harmonics.push_back(f);
            if(harmonics.size() >= N) {
                break;
            }
        }

        // calculate mismatch between predicted and actual peaks
        for(int i = 0; i < harmonics.size(); i++) {
            sample h = harmonics[i];
            int k = best_match(h, peak_freqs);
            sample f = peaks[k]->frequency;
            sample a = peaks[k]->amplitude;
            err_pm += (fabs(h - f) * pow(h, -p)) +
                      (((a / max_amp) * (q * fabs(h - f)) * (pow(h, -p) - r)));
        }

        // calculate the mismatch between actual and predicted peaks
        for(int i = 0; i < peaks.size(); i++) {
            sample f = peaks[i]->frequency;
            sample a = peaks[i]->amplitude;
            int k = best_match(f, harmonics);
            sample h = harmonics[k];
            err_mp += (fabs(f - h) * pow(f, -p)) +
                      ((a / max_amp) * (q * fabs(f - h)) * (pow(f, -p) - r));
        }

        // calculate the total error for f_current as a fundamental frequency
        err[f_current] = (err_pm / harmonics.size()) +
                         (rho * err_mp / peaks.size());

        f_current += f_step;
    }

    // return the value with the minimum total error
    sample best_freq = 0;
    sample min_error = 22050;
    for(std::map<sample, sample>::iterator i = err.begin(); i != err.end(); i++) {
        if(fabs((*i).second) < min_error) {
            min_error = fabs((*i).second);
            best_freq = (*i).first;
        }
    }

    return best_freq;
}
