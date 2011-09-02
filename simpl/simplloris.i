%module(directors="1") simplloris
%{
	#include "../src/simpl/simplloris.h"
%}
%include "base.i"

%feature("director") Simpl::LorisPeakDetection;
%include "../src/simpl/simplloris.h"
