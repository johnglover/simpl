 
////////////////////////////////////////////////////////////////////////
// This file is part of the SndObj library
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA 
//
// Copyright (c)Victor Lazzarini, 1997-2004
// See License.txt for a disclaimer of all warranties
// and licensing information

//////////////////////////////////////////////////////
//     IFGram.h: Instant Freq Analysis
//            
//          Victor Lazzarini, 2003
//           
/////////////////////////////////////////////////////////

#ifndef _IFGram_H
#define _IFGram_H
#include "PVA.h"

class IFGram : public PVA {

 protected:


  double* m_diffwin; // difference window
  double* m_fftdiff; // holds fft of diff window
  double* m_diffsig;
  double* m_pdiff;

 private:

  void inline IFAnalysis(double* signal); 

 public:

  IFGram();
  IFGram(Table* window, SndObj* input, double scale=1.f,
	 int fftsize=DEF_FFTSIZE, int hopsize=DEF_VECSIZE, double sr=DEF_SR);

  ~IFGram();
 
  int Set(char* mess, double value);
  int Connect(char* mess, void* input);
  void SetFFTSize(int fftsize);
  short DoProcess();
  
};

#endif





