 
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

#ifndef _RESYN_H
#define _RESYN_H

#include "SinSyn.h"

class ReSyn : public SinSyn {

 protected:

  float m_pitch;
  float m_tscal;

 public:

  ReSyn();
  ReSyn(SinAnal* input, int maxtracks, Table* table, float pitch=1.f, 
	float scale=1.f, float tscal=1.f, int vecsize=DEF_VECSIZE, 
	float sr=DEF_SR);
  void SetPitch(float pitch){ m_pitch = pitch; }
  void SetTimeScale(float scale) { m_tscal = scale; }
  int Set(char* mess, float value);
  ~ReSyn();
  short DoProcess();


};

#endif
