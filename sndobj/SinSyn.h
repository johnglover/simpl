 
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

#ifndef _SINSYN_H
#define _SINSYN_H

#include "SndObj.h"
#include "SinAnal.h"
#include "Table.h"

class SinSyn : public SndObj {

 protected:

  double m_size;
  Table* m_ptable;

  double m_factor;
  double m_facsqr;
  double m_LoTWOPI;
  double m_scale;
  double m_incr;
  double m_ratio;

  int m_tracks;
  int* m_trackID;
  int m_maxtracks;
 
  double* m_phases;
  double* m_freqs;
  double* m_amps;
 
 public:

  SinSyn();
  SinSyn(SinAnal* input, int maxtracks, Table* table, double scale=1.f, 
	 int vecsize=DEF_VECSIZE, double sr=DEF_SR);

  ~SinSyn();
  void SetTable(Table* table); 
  void SetMaxTracks(int maxtracks);
  void SetScale(double scale) { m_scale = scale; }
  int Set(char* mess, double value);
  int Connect(char* mess, void* input);
  short DoProcess();


};

#endif
