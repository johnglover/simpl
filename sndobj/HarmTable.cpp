 
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
//  HarmTable.cpp: implemenation of the HarmTable class       //
//                 (harmonic function table).                 //
//                                                            //
//                                                            //
//************************************************************//

#include "HarmTable.h"

/////////////// CONSTRUCTION / DESTRUCTION /////////////////////

HarmTable :: HarmTable(){

  m_L = 1024;
  m_harm = 1;
  m_typew = SINE;
  m_phase  = 0.f;
  m_table = new float[m_L+1];
  MakeTable();

}


HarmTable :: HarmTable(long L, int harm, int type, float phase){

  m_L = L;
  m_harm = harm;
  m_typew = type;
  m_phase = (float)(phase*TWOPI);

  m_table = new float [m_L+1];
  MakeTable();

}


HarmTable :: ~HarmTable(){

  delete[] m_table;

}


///////////// OPERATIONS ////////////////////////////////////

void
HarmTable::SetHarm(int harm, int type)
{
  m_harm = harm;
  m_typew = type;
  MakeTable();
}

short
HarmTable :: MakeTable(){
  
  float max = 1.f;	
  int n = 1, harm = m_harm, i;       

  switch (m_typew){
  case SINE:
    for(i=0; i < m_L; i++)
      m_table[i] = (float)(sin(i*TWOPI/m_L + m_phase)); 
    break;

  case SAW:
    ZeroTable();
    for(i=0; i < m_L; i++){
      for(n = 1 ; n <= harm ; n++)
	m_table[i] += (float)((1/(float)n)*sin(n*i*TWOPI/m_L + m_phase));
      max = (fabs((double)max) < fabs((double)m_table[i])) ? m_table[i] : max;
    }
    break;

  case SQUARE:
    ZeroTable();
    for(i=0; i < m_L; i++){
      for(n = 1 ; n <= harm ; n+=2)
	m_table[i] += (float)((1/(float)n)*sin(n*TWOPI*i/m_L + m_phase));		 
      max = (fabs((double)max) < fabs((double)m_table[i])) ? m_table[i] : max;
    }
    break;

  case BUZZ:
    ZeroTable();
    for(i=0; i < m_L; i++){
      for(n = 1 ; n <= harm ; n++)
	m_table[i] += (float) sin(n*TWOPI*i/m_L + m_phase);			
      max = (fabs((double)max) < fabs((double)m_table[i])) ? m_table[i] : max;
    }
    break;

  default:
    ZeroTable();
    m_error = 1;          
    return 0;
    break;
  } 

  //normalize:
  if(m_typew!=SINE)
    for(n = 0; n < m_L; n++)
      m_table[n] = m_table[n]/max;
  m_table[m_L] = m_table[0];  // guard point
  return 1;            
}

///////////////// ERROR HANDLING ///////////////////////////////

char*
HarmTable::ErrorMessage(){
  
  char* message;
   
  switch(m_error){

  case 0:
    message = "No error.";
    break; 

  case 1:
    message = "MakeTable() failed. Unsupported wave type.";
    break;

  default:
    message = "Undefined error";
    break;
  }

  return message;

}






