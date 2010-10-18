 
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

//************************************************************//
//  HarmTable.h: interface of the HarmTable class (harmonic   //
//           function table).                                 //
//                                                            //
//                                                            //
//************************************************************//
#ifndef _HARMTABLE_H 
#define _HARMTABLE_H

#include "Table.h"

enum { SINE=1, SAW, SQUARE, BUZZ };

class HarmTable : public Table {
  protected :

    int m_harm;
  float m_phase;
  int m_typew;
         
 public:
 
  void SetHarm(int harm, int type);

  char* ErrorMessage();
  short MakeTable();
  HarmTable();
  void SetPhase(float phase){ m_phase = (float)(phase*TWOPI); }
  HarmTable(long L, int harm, int type, float phase=0.f);
  ~HarmTable();
};


#endif








