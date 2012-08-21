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
//     PVA.h: Phase Vocoder Analysis Class
//            
//          Victor Lazzarini, 2003
//           
/////////////////////////////////////////////////////////

#ifndef _PVA_H 
#define _PVA_H
#include "FFT.h"

class PVA : public FFT {
 protected:
  int  m_rotcount; // rotation counter 
  double m_factor;  // conversion factor
  double* m_phases;

 private:
  void inline pvanalysis(double* signal); 

 public:
  PVA();
  PVA(Table* window, SndObj* input, double scale=1.f,
      int fftsize=DEF_FFTSIZE, int hopsize=DEF_VECSIZE, double sr=DEF_SR);
  ~PVA();

  double Outphases(int pos){ return m_phases[pos]; } // reads phase output.
  int Set(const char* mess, double value);
  void SetFFTSize(int fftsize);
  void SetHopSize(int hopsize);
  short DoProcess();
};

#endif
