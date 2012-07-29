 
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

#ifndef _SINANAL_H
#define _SINANAL_H

#include "SndObj.h"
#include "PVA.h"

class SinAnal : public SndObj {

 protected:

  double** m_bndx;  // bin indexes
  double** m_pkmags;  // peak mags
  double** m_adthresh;  // thresholds
  unsigned int** m_tstart;  // start times 
  unsigned int** m_lastpk; // end times
  unsigned int** m_trkid; // track ids

  double* m_phases; // phases
  double* m_freqs;  // frequencies
  double* m_mags;   // magnitudes
  double* m_bins;   // track bin indexes
  int* m_trndx;    // track IDs

  double* m_binmax;  // peak bin indexes
  double* m_magmax;  // peak mags
  double* m_diffs;    // differences

  int* m_maxix;     // max peak locations
  bool* m_contflag; // continuation flags

  int m_numbins;    // number of bins
  int m_maxtracks;   // max number of tracks
  double m_startupThresh;  // startup threshold
  double m_thresh;    // threshold

  int m_tracks;      // tracks in a frame
  int m_prev;         
  int m_cur;
  int m_accum;       // ID counter
  unsigned int m_timecount;
  int m_minpoints;     // minimun number of points in track
  int m_maxgap;     // max gap (in points) between consecutive points
  int m_numpeaks;  // number of peaks found in peak detection

 private:

  void sinanalysis();
  int peakdetection();

 public:

  SinAnal();
  SinAnal(SndObj* input, double threshold, int maxtracks, int minpoints=1,
	      int maxgap=3, double sr=DEF_SR);
  SinAnal(SndObj* input, int numbins, double threshold, int maxtracks, int minpoints=1,
  	      int maxgap=3, double sr=DEF_SR);
  ~SinAnal();

  virtual int GetTrackID(int track){ return m_trndx[track]; }
  virtual int GetTracks(){ return m_tracks; }

  int Set(const char* mess, double value);
  int Connect(const char* mess, void* input);

  void SetThreshold(double threshold){ m_thresh = threshold; }
  void SetIFGram(SndObj* input);
  void SetMaxTracks(int maxtracks);

  int FindPeaks();
  void SetPeaks(int numamps, double* amps, int numfreqs, double* freqs,
		        int numphases, double* phases);
  void PartialTracking();
  short DoProcess();
};

#endif
