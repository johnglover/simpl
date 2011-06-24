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
/*! \file fixTracks.c
 * \brief functions for making smoothly evolving tracks (partial frequencies)
 * 
 * Tries to fix gaps and short tracks
 */

#include "sms.h"

/*! \brief fill a gap in a given track 
 *
 * \param iCurrentFrame      currrent frame number 
 * \param iTrack                      track to be filled
 * \param pIState                 pointer to the state of tracks
 * \param pAnalParams       pointer to analysis parameters
 */
static void FillGap(int iCurrentFrame, int iTrack, int *pIState, 
                    SMS_AnalParams *pAnalParams)
{
	int iFrame, iLastFrame = - (pIState[iTrack] - 1);
	sfloat fConstant = TWO_PI / pAnalParams->iSamplingRate;
	sfloat fFirstMag, fFirstFreq, fLastMag, fLastFreq, fIncrMag, fIncrFreq,
		fMag, fTmpPha, fFreq;
  
	if(iCurrentFrame - iLastFrame < 0)
		return;
  
	/* if firstMag is 0 it means that there is no Gap, just the begining of a track */
	if(pAnalParams->ppFrames[iCurrentFrame - 
	   iLastFrame]->deterministic.pFSinAmp[iTrack] == 0)
	{
		pIState[iTrack] = 1;
		return;
	}
  
	fFirstMag = 
		pAnalParams->ppFrames[iCurrentFrame - iLastFrame]->deterministic.pFSinAmp[iTrack];
	fFirstFreq = 
		pAnalParams->ppFrames[iCurrentFrame - iLastFrame]->deterministic.pFSinFreq[iTrack];
	fLastMag = pAnalParams->ppFrames[iCurrentFrame]->deterministic.pFSinAmp[iTrack];
	fLastFreq = pAnalParams->ppFrames[iCurrentFrame]->deterministic.pFSinFreq[iTrack];
	fIncrMag =  (fLastMag - fFirstMag) / iLastFrame;
	fIncrFreq =  (fLastFreq - fFirstFreq) / iLastFrame;
  
	/* if inharmonic format and the two extremes are very different  */
	/* do not interpolate, it means that they are different tracks */
	if((pAnalParams->iFormat == SMS_FORMAT_IH ||
	    pAnalParams->iFormat == SMS_FORMAT_IHP) &&
	   (MIN (fFirstFreq, fLastFreq) * .5 * pAnalParams->fFreqDeviation <
	   fabs(fLastFreq - fFirstFreq)))
	{
		pIState[iTrack] = 1;
		return;		
	}

	fMag = fFirstMag;
	fFreq = fFirstFreq;
	/* fill the gap by interpolating values */
	/* if the gap is too long it should consider the lower partials */
	for(iFrame = iCurrentFrame - iLastFrame + 1; iFrame < iCurrentFrame; iFrame++)
	{
		/* interpolate magnitude */
		fMag += fIncrMag;
		pAnalParams->ppFrames[iFrame]->deterministic.pFSinAmp[iTrack] = fMag;
		/* interpolate frequency */
		fFreq += fIncrFreq;
		pAnalParams->ppFrames[iFrame]->deterministic.pFSinFreq[iTrack] = fFreq;
		/*interpolate phase (this may not be the right way) */
		fTmpPha = 
			pAnalParams->ppFrames[iFrame-1]->deterministic.pFSinPha[iTrack] -
				(pAnalParams->ppFrames[iFrame-1]->deterministic.pFSinFreq[iTrack] * 
				fConstant) * pAnalParams->sizeHop;
		pAnalParams->ppFrames[iFrame]->deterministic.pFSinPha[iTrack] = 
			fTmpPha - floor(fTmpPha/ TWO_PI) * TWO_PI;
	}
  
	if(pAnalParams->iDebugMode == SMS_DBG_CLEAN_TRAJ || 
	   pAnalParams->iDebugMode == SMS_DBG_ALL)
	{
		fprintf (stdout, "fillGap: track %d, frames %d to %d filled\n",
		        iTrack, pAnalParams->ppFrames[iCurrentFrame-iLastFrame + 1]->iFrameNum, 
		        pAnalParams->ppFrames[iCurrentFrame-1]->iFrameNum);
		fprintf (stdout, "firstFreq %f lastFreq %f, firstMag %f lastMag %f\n",
		        fFirstFreq, fLastFreq, fFirstMag, fLastMag);

  	}

	/* reset status */
	pIState[iTrack] = pAnalParams->iMinTrackLength;
}


/*! \brief delete a short track 
 *
 * this function is not exported to sms.h
 *
 * \param iCurrentFrame    current frame
 * \param iTrack                    track to be deleted
 * \param pIState               pointer to the state of tracks
 * \param pAnalParams     pointer to analysis parameters
 */
static void DeleteShortTrack(int iCurrentFrame, int iTrack, int *pIState,
                             SMS_AnalParams *pAnalParams)
{
	int iFrame, frame;
  
	for(iFrame = 1; iFrame <= pIState[iTrack]; iFrame++)
	{
		frame = iCurrentFrame - iFrame;
      
		if(frame <= 0)
			return;
      
		pAnalParams->ppFrames[frame]->deterministic.pFSinAmp[iTrack] = 0;
		pAnalParams->ppFrames[frame]->deterministic.pFSinFreq[iTrack] = 0;
		pAnalParams->ppFrames[frame]->deterministic.pFSinPha[iTrack] = 0;
	}
  
	if(pAnalParams->iDebugMode == SMS_DBG_CLEAN_TRAJ ||
	   pAnalParams->iDebugMode == SMS_DBG_ALL)
		fprintf(stdout, "deleteShortTrack: track %d, frames %d to %d deleted\n",
		        iTrack, pAnalParams->ppFrames[iCurrentFrame - pIState[iTrack]]->iFrameNum, 
		        pAnalParams->ppFrames[iCurrentFrame-1]->iFrameNum);
  
	/* reset state */
	pIState[iTrack] = -pAnalParams->iMaxSleepingTime;
}

/*! \brief fill gaps and delete short tracks 
 *
 * \param iCurrentFrame     current frame number
 * \param pAnalParams      pointer to analysis parameters
 */
void sms_cleanTracks(int iCurrentFrame, SMS_AnalParams *pAnalParams)
{
    int iTrack, iLength, iFrame;

    /* if fundamental and first partial are short, delete everything */
    if((pAnalParams->iFormat == SMS_FORMAT_H || pAnalParams->iFormat == SMS_FORMAT_HP) &&
       pAnalParams->ppFrames[iCurrentFrame]->deterministic.pFSinAmp[0] == 0 &&
       pAnalParams->guideStates[0] > 0 &&
       pAnalParams->guideStates[0] < pAnalParams->iMinTrackLength &&
       pAnalParams->ppFrames[iCurrentFrame]->deterministic.pFSinAmp[1] == 0 &&
       pAnalParams->guideStates[1] > 0 &&
       pAnalParams->guideStates[1] < pAnalParams->iMinTrackLength)
    {
        iLength = pAnalParams->guideStates[0];
        for(iTrack = 0; iTrack < pAnalParams->nGuides; iTrack++)
        {
            for(iFrame = 1; iFrame <= iLength; iFrame++)
            {
                if((iCurrentFrame - iFrame) >= 0)
                {
                    pAnalParams->ppFrames[iCurrentFrame - 
                        iFrame]->deterministic.pFSinAmp[iTrack] = 0;
                    pAnalParams->ppFrames[iCurrentFrame - 
                        iFrame]->deterministic.pFSinFreq[iTrack] = 0;
                    pAnalParams->ppFrames[iCurrentFrame - 
                        iFrame]->deterministic.pFSinPha[iTrack] = 0;
                }
            }
            pAnalParams->guideStates[iTrack] = -pAnalParams->iMaxSleepingTime;
        }
        if(pAnalParams->iDebugMode == SMS_DBG_CLEAN_TRAJ || 
           pAnalParams->iDebugMode == SMS_DBG_ALL)
        {
            fprintf(stdout, "cleanTrack: frame %d to frame %d deleted\n",
                    pAnalParams->ppFrames[iCurrentFrame-iLength]->iFrameNum, 
                    pAnalParams->ppFrames[iCurrentFrame-1]->iFrameNum);
        }

        return;
    }

    /* check every partial individually */
    for(iTrack = 0; iTrack < pAnalParams->nGuides; iTrack++)
    {
        /* track after gap */
        if(pAnalParams->ppFrames[iCurrentFrame]->deterministic.pFSinAmp[iTrack] != 0)
        { 
            if(pAnalParams->guideStates[iTrack] < 0 && 
               pAnalParams->guideStates[iTrack] > -pAnalParams->iMaxSleepingTime)
                FillGap (iCurrentFrame, iTrack, pAnalParams->guideStates, pAnalParams);
            else
                pAnalParams->guideStates[iTrack] = 
                    (pAnalParams->guideStates[iTrack]<0) ? 1 : pAnalParams->guideStates[iTrack]+1;
        }
        /* gap after track */
        else
        {      
            if(pAnalParams->guideStates[iTrack] > 0 &&  
               pAnalParams->guideStates[iTrack] < pAnalParams->iMinTrackLength)
                DeleteShortTrack (iCurrentFrame, iTrack, pAnalParams->guideStates, pAnalParams);
            else 
                pAnalParams->guideStates[iTrack] =
                    (pAnalParams->guideStates[iTrack]>0) ? -1 : pAnalParams->guideStates[iTrack]-1;
        }
    }
    return;
}

/*! \brief scale deterministic magnitude if synthesis is larger than original 
 *
 * \param pFSynthBuffer      synthesis buffer
 * \param pFOriginalBuffer   original sound
 * \param pFSinAmp          magnitudes to be scaled
 * \param pAnalParams      pointer to analysis parameters
 * \param nTrack                    number of tracks
 */
void sms_scaleDet(sfloat *pFSynthBuffer, sfloat *pFOriginalBuffer, 
                  sfloat *pFSinAmp, SMS_AnalParams *pAnalParams, int nTrack)
{
	sfloat fOriginalMag = 0, fSynthesisMag = 0;
	sfloat fCosScaleFactor;
	int iTrack, i;
  
	/* get sound energy */
	for(i = 0; i < pAnalParams->sizeHop; i++)
	{
		fOriginalMag += fabs(pFOriginalBuffer[i]); 
		fSynthesisMag += fabs(pFSynthBuffer[i]);
	}
  
	/* if total energy of deterministic sound is larger than original,
	   scale deterministic representation */
	if(fSynthesisMag > (1.5 * fOriginalMag))
	{
		fCosScaleFactor = fOriginalMag / fSynthesisMag;
      
		if(pAnalParams->iDebugMode == SMS_DBG_CLEAN_TRAJ || 
		   pAnalParams->iDebugMode == SMS_DBG_ALL)
			fprintf(stdout, "Frame %d: magnitude scaled by %f\n",
			        pAnalParams->ppFrames[0]->iFrameNum, fCosScaleFactor);
      
		for(iTrack = 0; iTrack < nTrack; iTrack++)
			if(pFSinAmp[iTrack] > 0)
				pFSinAmp[iTrack] = sms_magToDB(sms_dBToMag(pFSinAmp[iTrack]) * fCosScaleFactor);
	}
}

