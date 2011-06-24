%module(directors="1") simplsndobj
%{
	#include "../src/sndobj/SndObj.h"
	#include "../src/sndobj/SndIO.h"
	#include "../src/sndobj/Table.h"
	#include "../src/sndobj/FFT.h"
	#include "../src/sndobj/IFFT.h"
	#include "../src/sndobj/PVA.h"
	#include "../src/sndobj/PVS.h"
	#include "../src/sndobj/IFGram.h"
	#include "../src/sndobj/SinAnal.h"
	#include "../src/sndobj/SinSyn.h"
	#include "../src/sndobj/AdSyn.h"
	#include "../src/sndobj/ReSyn.h"
	#include "../src/sndobj/HarmTable.h"
	#include "../src/sndobj/HammingTable.h"
	#define SWIG_FILE_WITH_INIT
%}

%feature("director") SndObj;
%feature("director") SinAnal;

%include "numpy.i"

%init 
%{
    import_array();
%}

%ignore SndObj::SndObj(SndObj &);
%ignore SndObj::operator=(SndObj);

%apply(double* IN_ARRAY1, int DIM1) {(double* in_vector, int size)};
%apply(double* INPLACE_ARRAY1, int DIM1) {(double* out_vector, int size)};
%include "../src/sndobj/SndObj.h"
%clear(double* in_vector, int size);
%clear(double* out_vector, int size);

%include "../src/sndobj/SndIO.h"
%include "../src/sndobj/Table.h"
%include "../src/sndobj/FFT.h"
%include "../src/sndobj/IFFT.h"
%include "../src/sndobj/PVA.h"
%include "../src/sndobj/PVS.h"
%include "../src/sndobj/IFGram.h"

%apply (int DIM1, double* IN_ARRAY1)
{
    (int numamps, double* amps),
    (int numfreqs, double* freqs),
    (int numphases, double* phases)
}
%include "../src/sndobj/SinAnal.h"
%include "../src/sndobj/SinSyn.h"
%include "../src/sndobj/ReSyn.h"
%include "../src/sndobj/AdSyn.h"
%include "../src/sndobj/HarmTable.h"
%include "../src/sndobj/HammingTable.h"
