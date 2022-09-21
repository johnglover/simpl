#ifndef SIMPL_TWM_H
#define SIMPL_TWM_H

#include <map>
#include <vector>

#include "../simpl/base.h"

namespace simpl
{

int best_match(simpl_sample freq, std::vector<simpl_sample> candidates);

simpl_sample twm(Peaks peaks, simpl_sample f_min=20.0,
           simpl_sample f_max=3000.0, simpl_sample f_step=10.0);

}

#endif
