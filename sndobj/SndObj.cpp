 
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

///////////////////////////////////////
// SndObj Library Version 2
// SndObj class
// a very basic model for all derived classes
//
//////////////////////////////////////
#include "SndObj.h"
#include "SndIO.h"

SndObj::SndObj(){ 
  m_output = NULL;
  SetVectorSize(DEF_VECSIZE);
  m_input = 0;
  m_sr = DEF_SR;
  m_error =0;
  for(m_vecpos = 0; m_vecpos < m_vecsize; m_vecpos++) 
    m_output[m_vecpos]= 0.f;
  m_msgtable = new msg_link;
  m_msgtable->previous = 0;
  AddMsg("SR", 1);
  AddMsg("vector size", 2);
  AddMsg("input", 3);
  Enable();
	
}

SndObj::SndObj(SndObj* input, int vecsize, float sr){
  m_output = NULL;
  SetVectorSize(vecsize);	
  m_input = input; 
  m_sr = sr;
  m_error = 0;
  for(m_vecpos = 0; m_vecpos < m_vecsize; m_vecpos++) m_output[m_vecpos]= 0.f;
	
  m_msgtable = new msg_link;
  m_msgtable->previous = 0;
  AddMsg("SR", 1);
  AddMsg("vector size", 2);
  AddMsg("input", 3);
  Enable();
	
}	


SndObj::SndObj(SndObj& obj){
  m_output = NULL;
  SetVectorSize(obj.GetVectorSize());
  SetSr(obj.GetSr());
	
  for(int n=0; n<m_vecsize;n++)m_output[n]=obj.Output(n);
  m_input = obj.GetInput();
  m_error =0;

  m_msgtable = new msg_link;
  m_msgtable->previous = 0;
  AddMsg("SR", 1);
  AddMsg("vector size", 2);
  AddMsg("input", 3);
  Enable();
	
}


SndObj::~SndObj(){

  delete[] m_output;

  msg_link *todestroy = m_msgtable;
  while(m_msgtable->previous){
    m_msgtable = todestroy->previous;
    delete todestroy;
    todestroy = m_msgtable;
  }
  delete m_msgtable;

}

void
SndObj::GetMsgList(string* list){
  msg_link* tmp = m_msgtable;
  while(tmp->previous){
    list->append(tmp->msg);
    list->append("\n");
    tmp = tmp->previous;
  }
}


void
SndObj::AddMsg(const char* mess, int ID){

  msg_link* newlink = new msg_link;
  msg_link* tmp = m_msgtable;
  newlink->msg = mess;
  newlink->ID = ID; 
  m_msgtable = newlink;
  m_msgtable->previous = tmp;
}

int
SndObj::Connect(char* mess, void *input){

  switch (FindMsg(mess)){

  case 3:
    m_input = (SndObj *) input;
    return 1;

  default:
    return 0;

  }
}

int 
SndObj::Set(char* mess, float value){

  switch (FindMsg(mess)){

  case 1:
    SetSr(value);
    return 1;

  case 2:
    SetVectorSize((int) value);
    return 1;

  default:
    return 0;
     
  }


}


void
SndObj::SetVectorSize(int vecsize){
  if(m_output) delete[] m_output;
  if(!(m_output = new float[vecsize])){
    m_error = 1;
#ifdef DEBUG
    cout << ErrorMessage();
#endif
    m_vecsize = m_vecsize_max = 0;
    return;
  }
  m_vecsize = vecsize;
  m_vecsize_max = vecsize;
  m_altvecpos = m_vecpos = 0;
}

short
SndObj::DoProcess(){

  if(!m_error){
    if(m_input){
      for(m_vecpos = 0; m_vecpos < m_vecsize; m_vecpos++) {
	if(m_enable) m_output[m_vecpos] = m_input->Output(m_vecpos);
	else m_output[m_vecpos] = 0.f;
      } 
      return 1;
    } else {
      return 0;
    }
  }
  else return 0;
}

void 
SndObj::operator>>(SndIO& out){
  out.SetOutput(1, this);
  out.Write();
}

void 
SndObj::operator<<(SndIO& in){
  in.Read();
  for(int n=0;n<m_vecsize;n++) m_output[n]=in.Output(n,1);
}


char* SndObj::ErrorMessage(){
	 
  char* message;
   
  switch(m_error){

  case 0:
    message = "No error\n";
    break; 

  case 1:
    message = "Failed to allocate vector memory\n";
    break;

  case 3:
    message = "DoProcess() failed: no input object \n";
    break;

  default:
    message = "Undefined error\n";
    break;
  
  }

  return message;

}

const char* SndObj::CErrorMessage(){
	   
  switch(m_error){

  case 0:
    return "No error\n";
    break; 

  case 1:
    return "Failed to allocate vector memory\n";
    break;

  case 3:
    return "DoProcess() failed: no input object \n";
    break;

  default:
    return "Undefined error\n";
  
  }

}
