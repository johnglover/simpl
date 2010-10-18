%module(directors="1") pysndobj
%{
	#include "SndObj.h"
	#include "SndIO.h"
	#include "Table.h"
	#include "FFT.h"
	#include "IFFT.h"
	#include "PVA.h"
	#include "PVS.h"
	#include "IFGram.h"
	#include "SinAnal.h"
	#include "SinSyn.h"
	#include "AdSyn.h"
	#include "ReSyn.h"
	#include "HarmTable.h"
	#include "HammingTable.h"
	#define SWIG_FILE_WITH_INIT
%}

%feature("director") SndObj;
%feature("director") SinAnal;

%include "../common/numpy.i"

%init 
%{
    import_array();
%}

%ignore SndObj::SndObj(SndObj &);
%ignore SndObj::operator=(SndObj);

%apply(float* IN_ARRAY1, int DIM1) {(float* in_vector, int size)};
%apply(float* INPLACE_ARRAY1, int DIM1) {(float* out_vector, int size)};
%include "SndObj.h"
%clear(float* in_vector, int size);
%clear(float* out_vector, int size);

%include "SndIO.h"
%include "Table.h"
%include "FFT.h"
%include "IFFT.h"
%include "PVA.h"
%include "PVS.h"
%include "IFGram.h"

%apply (int DIM1, float* IN_ARRAY1)
{
    (int numamps, float* amps),
    (int numfreqs, float* freqs),
    (int numphases, float* phases)
}
%include "SinAnal.h"
%include "SinSyn.h"
%include "ReSyn.h"
%include "AdSyn.h"
%include "HarmTable.h"
%include "HammingTable.h"
