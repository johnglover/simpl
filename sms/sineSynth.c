/* 
 * Copyright (c) 2008 MUSIC TECHNOLOGY GROUP (MTG)
 *                         UNIVERSITAT POMPEU FABRA 
 * 
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 * 
 */
/*! \file sineSynth.c
 * \brief functions for synthesizing evolving sinusoids
 */

#include "sms.h"

/*! \brief generate a sinusoid given two peaks, current and last
 *
 * it interpolation between phase values and magnitudes   
 *
 * \param fFreq                    current frequency
 * \param fMag                    current magnitude
 * \param fPhase                 current phase
 * \param pLastFrame         stucture with values from last frame
 * \param pFWaveform        pointer to output waveform
 * \param sizeBuffer        size of the synthesis buffer 
 * \param iTrack                  current track 
 */
static void SinePhaSynth(sfloat fFreq, sfloat fMag, sfloat fPhase,
                         SMS_Data *pLastFrame, sfloat *pFWaveform, 
                         int sizeBuffer, int iTrack)
{
    sfloat  fMagIncr, fInstMag, fInstPhase, fTmp;
    int iM, i;
    sfloat fAlpha, fBeta, fTmp1, fTmp2;

    /* if no mag in last frame copy freq from current and make phase */
    if (pLastFrame->pFSinAmp[iTrack] <= 0)
    {
        pLastFrame->pFSinFreq[iTrack] = fFreq;
        fTmp = fPhase - (fFreq * sizeBuffer);
        pLastFrame->pFSinPha[iTrack] = fTmp - floor(fTmp / TWO_PI) * TWO_PI;
    }
    /* and the other way */
    else if (fMag <= 0)
    {
        fFreq = pLastFrame->pFSinFreq[iTrack];
        fTmp = pLastFrame->pFSinPha[iTrack] + 
            (pLastFrame->pFSinFreq[iTrack] * sizeBuffer);
        fPhase = fTmp - floor(fTmp / TWO_PI) * TWO_PI;
    }

    /* caculate the instantaneous amplitude */
    fMagIncr = (fMag - pLastFrame->pFSinAmp[iTrack]) / sizeBuffer;
    fInstMag = pLastFrame->pFSinAmp[iTrack];

    /* create instantaneous phase from freq. and phase values */
    fTmp1 = fFreq - pLastFrame->pFSinFreq[iTrack];
    fTmp2 = ((pLastFrame->pFSinPha[iTrack] + 
                pLastFrame->pFSinFreq[iTrack] * sizeBuffer - fPhase) +
            fTmp1 * sizeBuffer / 2.0) / TWO_PI;
    iM = (int) (fTmp2 + .5);
    fTmp2 = fPhase - pLastFrame->pFSinPha[iTrack] - 
        pLastFrame->pFSinFreq[iTrack] * sizeBuffer +
        TWO_PI * iM;
    fAlpha = (3.0 / (sfloat)(sizeBuffer * sizeBuffer)) * 
        fTmp2 - fTmp1 / sizeBuffer;
    fBeta = (-2.0 / ((sfloat) (sizeBuffer * sizeBuffer * sizeBuffer))) * 
        fTmp2 + fTmp1 / ((sfloat) (sizeBuffer * sizeBuffer));

    for(i=0; i<sizeBuffer; i++)
    {
        fInstMag += fMagIncr;
        fInstPhase = pLastFrame->pFSinPha[iTrack] + 
            pLastFrame->pFSinFreq[iTrack] * i + 
            fAlpha * i * i + fBeta * i * i * i;

        /*     pFWaveform[i] += sms_dBToMag(fInstMag) * sms_sine(fInstPhase + PI_2); */
        pFWaveform[i] += sms_dBToMag(fInstMag) * sinf(fInstPhase + PI_2);
    }
    /* save current values into buffer */
    pLastFrame->pFSinFreq[iTrack] = fFreq;
    pLastFrame->pFSinAmp[iTrack] = fMag;
    pLastFrame->pFSinPha[iTrack] = fPhase;
}

/*! \brief generate a sinusoid given two frames, current and last
 * 
 * \param fFreq          current frequency 
 * \param fMag                 current magnitude  
 * \param pLastFrame      stucture with values from last frame 
 * \param pFBuffer           pointer to output waveform 
 * \param sizeBuffer     size of the synthesis buffer 
 * \param iTrack               current track 
 */
static void SineSynth(sfloat fFreq, sfloat fMag, SMS_Data *pLastFrame,
                      sfloat *pFBuffer, int sizeBuffer, int iTrack)
{
    sfloat  fMagIncr, fInstMag, fFreqIncr, fInstPhase, fInstFreq;
    int i;

    /* if no mag in last frame copy freq from current */
    if (pLastFrame->pFSinAmp[iTrack] <= 0)
    {
        pLastFrame->pFSinFreq[iTrack] = fFreq;
        pLastFrame->pFSinPha[iTrack] = 
            TWO_PI * sms_random();
    }
    /* and the other way */
    else if (fMag <= 0)
        fFreq = pLastFrame->pFSinFreq[iTrack];

    /* calculate the instantaneous amplitude */
    fMagIncr = (fMag - pLastFrame->pFSinAmp[iTrack]) / sizeBuffer;
    fInstMag = pLastFrame->pFSinAmp[iTrack];
    /* calculate instantaneous frequency */
    fFreqIncr = (fFreq - pLastFrame->pFSinFreq[iTrack]) / sizeBuffer;
    fInstFreq = pLastFrame->pFSinFreq[iTrack];
    fInstPhase = pLastFrame->pFSinPha[iTrack];

    /* generate all the samples */    
    for (i = 0; i < sizeBuffer; i++)
    {
        fInstMag += fMagIncr;
        fInstFreq += fFreqIncr;
        fInstPhase += fInstFreq;

        pFBuffer[i] += sms_dBToMag(fInstMag) * sms_sine(fInstPhase);
    }

    /* save current values into last values */
    pLastFrame->pFSinFreq[iTrack] = fFreq;
    pLastFrame->pFSinAmp[iTrack] = fMag;
    pLastFrame->pFSinPha[iTrack] = fInstPhase - 
        floor(fInstPhase / TWO_PI) * TWO_PI;
}

/*! \brief generate all the sinusoids for a given frame
 * 
 * \param pSmsData       SMS data for current frame 
 * \param pFBuffer         pointer to output waveform 
 * \param sizeBuffer        size of the synthesis buffer
 * \param pLastFrame    SMS data from last frame 
 * \param iSamplingRate sampling rate to synthesize for
 */
void sms_sineSynthFrame(SMS_Data *pSmsData, sfloat *pFBuffer, 
                        int sizeBuffer, SMS_Data *pLastFrame, 
                        int iSamplingRate)
{
    sfloat fMag, fFreq;
    int i;
    int nTracks = pSmsData->nTracks;
    int iHalfSamplingRate = iSamplingRate >> 1;

    /* go through all the tracks */    
    for (i = 0; i < nTracks; i++)
    {
        /* get magnitude */
        fMag = pSmsData->pFSinAmp[i];

        fFreq = pSmsData->pFSinFreq[i];

        /* gaurd so transposed frequencies don't alias */
        if (fFreq > iHalfSamplingRate || fFreq < 0) 
            fMag = 0;

        /* generate sines if there are magnitude values */
        if ((fMag > 0) || (pLastFrame->pFSinAmp[i] > 0))
        {  
            /* frequency from Hz to radians */
            fFreq = (fFreq == 0) ? 0 : TWO_PI * fFreq / iSamplingRate;

            /* \todo make seperate function for SineSynth /wo phase */
            if (pSmsData->pFSinPha == NULL)
            {                
                SineSynth(fFreq, fMag, pLastFrame, pFBuffer, sizeBuffer, i);
            }
            else
            {
                SinePhaSynth(fFreq, fMag, pSmsData->pFSinPha[i], pLastFrame, 
                             pFBuffer, sizeBuffer, i);
            }
        }
    }
}     

