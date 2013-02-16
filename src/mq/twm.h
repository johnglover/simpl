#ifndef SIMPL_TWM_H
#define SIMPL_TWM_H

#include <map>
#include <vector>

#include "base.h"

namespace simpl
{

int best_match(sample freq, std::vector<sample> candidates);

sample twm(Peaks peaks, sample f_min=20.0,
           sample f_max=3000.0, sample f_step=10.0);

}

#endif
