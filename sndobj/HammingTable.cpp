 
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
//  HammingTable.cpp:  implementation of the HammingTable     //
//   object (Generalized Hamming Window function table)       //
//                                                            //
//                                                            //
//************************************************************//
#include "HammingTable.h"
//////////construction / destruction ///////////////////////
HammingTable :: HammingTable(){

  m_L = 1024;
  m_alpha = .54f;
  m_table = new float[m_L+1];
  MakeTable();

}

HammingTable :: HammingTable(long L, float alpha){

  m_L = L;
  m_alpha = alpha;
  m_table = new float [m_L+1];
  MakeTable();

}


HammingTable :: ~HammingTable(){

  delete[] m_table;

}


///////////// OPERATIONS ////////////////////////////////////
void 
HammingTable :: SetParam(long L, float alpha){
           
  m_alpha = alpha;
  m_L = L;
  delete[] m_table;
  m_table = new float[m_L+1];
}

short
HammingTable :: MakeTable(){
  for(long n = 0; n < m_L; n++)
    m_table[n]= (float)(m_alpha - (1-m_alpha)*
			cos(n*TWOPI/(m_L-1.)));
  m_table[m_L] = m_table[m_L-1];   
  return 1;            
      

}

///////////////// ERROR HANDLING ///////////////////////////////

char*
HammingTable::ErrorMessage(){
  
  char* message;
   
  switch(m_error){

  case 0:
    message = "No error.";
    break; 


  default:
    message = "Undefined error";
    break;
  }

  return message;

}
