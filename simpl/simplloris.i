%module(directors="1") simplloris
%{
	#include "../src/simpl/base.h"
	#include "../src/simpl/exceptions.h"
	#include "../src/simpl/simplloris.h"
	#define SWIG_FILE_WITH_INIT
%}

%include "common.i"

%feature("director") Simpl::LorisPeakDetection;

%init 
%{
    import_array();
%}

%include "../src/simpl/base.h"
%include "../src/simpl/exceptions.h"
%include "../src/simpl/simplloris.h"
