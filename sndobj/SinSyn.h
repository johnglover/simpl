 
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

  float m_size;
  Table* m_ptable;

  float m_factor;
  float m_facsqr;
  float m_LoTWOPI;
  float m_scale;
  float m_incr;
  float m_ratio;

  int m_tracks;
  int* m_trackID;
  int m_maxtracks;
 
  float* m_phases;
  float* m_freqs;
  float* m_amps;
 
 public:

  SinSyn();
  SinSyn(SinAnal* input, int maxtracks, Table* table, float scale=1.f, 
	 int vecsize=DEF_VECSIZE, float sr=DEF_SR);

  ~SinSyn();
  void SetTable(Table* table); 
  void SetMaxTracks(int maxtracks);
  void SetScale(float scale) { m_scale = scale; }
  int Set(char* mess, float value);
  int Connect(char* mess, void* input);
  short DoProcess();


};

#endif
