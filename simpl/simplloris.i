%module(directors="1") simplloris
%{
	#include "../src/simpl/base.h"
	#include "../src/simpl/exceptions.h"
	#include "../src/simpl/simplloris.h"
	#define SWIG_FILE_WITH_INIT
%}

%include "common.i"

%feature("director") Simpl::Peak;
%feature("director") Simpl::LorisPeakDetection;

%init 
%{
    import_array();
%}

/* %apply(double* IN_ARRAY1, int DIM1) {(double* in_vector, int size)}; */
/* %apply(double* INPLACE_ARRAY1, int DIM1) {(double* out_vector, int size)}; */

/* %apply (int DIM1, double* IN_ARRAY1) */
/* { */
/*     (int numamps, double* amps), */
/*     (int numfreqs, double* freqs), */
/*     (int numphases, double* phases) */
/* } */
%include "../src/simpl/base.h"
%include "../src/simpl/exceptions.h"
%include "../src/simpl/simplloris.h"
