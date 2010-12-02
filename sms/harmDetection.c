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
/*! \file harmDetection.c
 * \brief Detection of a given harmonic
 */

#include "sms.h"

#define N_FUND_HARM 6       /*!< number of harmonics to use for fundamental detection */
#define N_HARM_PEAKS 4      /*!< number of peaks to check as possible ref harmonics */
#define FREQ_DEV_THRES .07  /*!< threshold for deviation from perfect harmonics */
#define MAG_PERC_THRES .6   /*!< threshold for magnitude of harmonics
                                 with respect to the total magnitude */
#define HARM_RATIO_THRES .8 /*!< threshold for percentage of harmonics found */

/*! \brief get closest peak to a given harmonic of the possible fundamental
 *  
 * \param iPeakCandidate     peak number of possible fundamental
 * \param nHarm              number of harmonic
 * \param pSpectralPeaks   pointer to all the peaks
 * \param pICurrentPeak     pointer to the last peak taken
 * \param iRefHarmonic    reference harmonic number
 * \return the number of the closest peak or -1 if not found  
 */
static int GetClosestPeak(int iPeakCandidate, int nHarm, SMS_Peak *pSpectralPeaks,
                          int *pICurrentPeak, int iRefHarmonic, int maxPeaks)
{
    int iBestPeak = *pICurrentPeak + 1;
    int iNextPeak = iBestPeak + 1;

    if((iBestPeak >= maxPeaks) || (iNextPeak >= maxPeaks))
        return -1;

    sfloat fBestPeakFreq = pSpectralPeaks[iBestPeak].fFreq,
           fHarmFreq = (1 + nHarm) * pSpectralPeaks[iPeakCandidate].fFreq / iRefHarmonic, 
           fMinDistance = fabs(fHarmFreq - fBestPeakFreq),
           fMaxPeakDev = .5 * fHarmFreq / (nHarm + 1), 
           fDistance = 0.0;
  
    fDistance = fabs(fHarmFreq - pSpectralPeaks[iNextPeak].fFreq);
    while((fDistance < fMinDistance) && (iNextPeak < maxPeaks - 1))
    {
        iBestPeak = iNextPeak;
        fMinDistance = fDistance;
        iNextPeak++;
        fDistance = fabs(fHarmFreq - pSpectralPeaks[iNextPeak].fFreq);
    }
  
    /* make sure the chosen peak is good */
    fBestPeakFreq = pSpectralPeaks[iBestPeak].fFreq;

    /* if best peak is not in the range */
    if(fabs(fBestPeakFreq - fHarmFreq) > fMaxPeakDev)
        return -1;
  
    *pICurrentPeak = iBestPeak;
    return iBestPeak;
}

/*! \brief checks if peak is substantial
 *
 *  check if peak is larger enough to be considered a fundamental
 *  without any further testing or too small to be considered
 *

 * \param fRefHarmMag      magnitude of possible fundamental
 * \param pSpectralPeaks   all the peaks
 * \param nCand              number of existing candidates
 * \param fRefHarmMagDiffFromMax value to judge the peak based on the difference of its magnitude compared to the reference
 * \return 1 if big peak, -1 if too small , otherwise return 0 
 */
static int ComparePeak(sfloat fRefHarmMag, SMS_Peak *pSpectralPeaks, int nCand, 
                       sfloat fRefHarmMagDiffFromMax, int maxPeaks)
{
    int iPeak;
    sfloat fMag = 0;
  
    /* if peak is very large take it as possible fundamental */
    if(nCand == 0 && fRefHarmMag > 80.)
        return 1;
  
    /* compare the peak with the first N_FUND_HARM peaks */
    /* if too small forget it */
    for(iPeak = 0; (iPeak < N_FUND_HARM) && (iPeak < maxPeaks); iPeak++)
    {
        if(pSpectralPeaks[iPeak].fMag > 0 &&
           fRefHarmMag - pSpectralPeaks[iPeak].fMag < - fRefHarmMagDiffFromMax)
            return -1;
    }
  
    /* if it is much bigger than rest take it */
    for(iPeak = 0; (iPeak < N_FUND_HARM) && (iPeak < maxPeaks); iPeak++)
    {
        fMag = pSpectralPeaks[iPeak].fMag;
        if(fMag <= 0 ||
           ((fMag != fRefHarmMag) && (nCand > 0) && (fRefHarmMag - fMag < 30.0)) ||
            ((nCand == 0) && (fRefHarmMag - fMag < 15.0)))
            return 0;
    }
    return 1;
}


/*! \brief check if the current peak is a harmonic of one of the candidates
 *               
 * \param fFundFreq          frequency of peak to be tested
 * \param pCHarmonic       all candidates accepted
 * \param nCand                location of las candidate
 * \return 1 if it is a harmonic, 0 if it is not    
 */
static int CheckIfHarmonic(sfloat fFundFreq, SMS_HarmCandidate *pCHarmonic, int nCand)
{
    int iPeak;
  
    /* go through all the candidates checking if they are fundamentals
     * of the peak to be considered */
    for(iPeak = 0; iPeak < nCand; iPeak++)
    {
        if(fabs(floor((double)(fFundFreq / pCHarmonic[iPeak].fFreq) + .5) -
                (fFundFreq / pCHarmonic[iPeak].fFreq)) <= .1)
            return 1;
    }
    return 0;
}


/*! \brief consider a peak as a possible candidate and give it a weight value, 
 *
 * \param iPeak                iPeak number to be considered
 * \param pSpectralPeaks     all the peaks
 * \param pCHarmonic  all the candidates
 * \param nCand                 candidate number that is to be filled
 * \param pPeakParams    analysis parameters
 * \param fRefFundamental     previous fundamental
 * \return -1 if not good enough for a candidate, return 0 if reached
 * the top frequency boundary, return -2 if stop checking because it 
 * found a really good one, return 1 if the peak is a good candidate 
 */

static int GoodCandidate(int iPeak, int maxPeaks, SMS_Peak *pSpectralPeaks, 
                         SMS_HarmCandidate *pCHarmonic, int nCand, int soundType, sfloat fRefFundamental,
                         sfloat minRefHarmMag, sfloat refHarmMagDiffFromMax, sfloat refHarmonic)
{
    sfloat fHarmFreq = 0.0, 
           fRefHarmFreq = 0.0, 
           fRefHarmMag = 0.0, 
           fTotalMag = 0.0, 
           fTotalDev = 0.0,
           fTotalMaxMag = 0.0, 
           fAvgMag = 0.0, 
           fAvgDev = 0.0, 
           fHarmRatio = 0.0;
    int iHarm = 0, 
        iChosenPeak = 0, 
        iPeakComp = 0, 
        iCurrentPeak = 0, 
        nGoodHarm = 0, 
        i = 0;

    fRefHarmFreq = fHarmFreq = pSpectralPeaks[iPeak].fFreq;

    fTotalDev = 0;
    fRefHarmMag = pSpectralPeaks[iPeak].fMag;
    fTotalMag = fRefHarmMag;

    /* check if magnitude is big enough */
    /*! \bug sfloat comparison to 0 */
    if(((fRefFundamental > 0) && (fRefHarmMag < minRefHarmMag - 10)) ||
       ((fRefFundamental <= 0) && (fRefHarmMag < minRefHarmMag)))
        return -1;

    /* check that it is not a harmonic of a previous candidate */
    if(nCand > 0 &&
       CheckIfHarmonic(fRefHarmFreq / refHarmonic, pCHarmonic, nCand))
        return -1;

    /* check if it is very big or very small */
    iPeakComp = ComparePeak(fRefHarmMag, pSpectralPeaks, nCand, refHarmMagDiffFromMax, maxPeaks);

    /* too small */
    if(iPeakComp == -1)
        return -1;
    /* very big */
    else if(iPeakComp == 1)
    {
        pCHarmonic[nCand].fFreq = fRefHarmFreq;
        pCHarmonic[nCand].fMag = fRefHarmMag;
        pCHarmonic[nCand].fMagPerc = 1;
        pCHarmonic[nCand].fFreqDev = 0;
        pCHarmonic[nCand].fHarmRatio = 1;
        return -2;
    }

    /* get a weight on the peak by comparing its harmonic series   */
    /* with the existing peaks */
    if(soundType != SMS_SOUND_TYPE_NOTE)
    {
        fHarmFreq = fRefHarmFreq;
        iCurrentPeak = iPeak;
        nGoodHarm = 0;
        for(iHarm = refHarmonic; (iHarm < N_FUND_HARM) && (iHarm < maxPeaks); iHarm++)
        {
            fHarmFreq += fRefHarmFreq / refHarmonic;
            iChosenPeak = GetClosestPeak(iPeak, iHarm, pSpectralPeaks,
                                         &iCurrentPeak, refHarmonic,
                                         maxPeaks);
            if(iChosenPeak > 0)
            {
                fTotalDev += fabs(fHarmFreq - pSpectralPeaks[iChosenPeak].fFreq) /
                                  fHarmFreq;
                fTotalMag += pSpectralPeaks[iChosenPeak].fMag;
                nGoodHarm++;
            }
        }

        for(i = 0; i <= iCurrentPeak; i++)
            fTotalMaxMag +=  pSpectralPeaks[i].fMag;

        fAvgDev = fTotalDev / (iHarm + 1);
        fAvgMag = fTotalMag / fTotalMaxMag;
        fHarmRatio = (sfloat)nGoodHarm / (N_FUND_HARM - 1);

        if(fRefFundamental > 0)
        {
            if(fAvgDev > FREQ_DEV_THRES || fAvgMag < MAG_PERC_THRES - .1 ||
               fHarmRatio < HARM_RATIO_THRES - .1)
                return -1;
        }
        else
        {
            if(fAvgDev > FREQ_DEV_THRES || fAvgMag < MAG_PERC_THRES ||
               fHarmRatio < HARM_RATIO_THRES)
                return -1;
        }
    }

    pCHarmonic[nCand].fFreq = fRefHarmFreq;
    pCHarmonic[nCand].fMag = fRefHarmMag;
    pCHarmonic[nCand].fMagPerc = fAvgMag;
    pCHarmonic[nCand].fFreqDev = fAvgDev;
    pCHarmonic[nCand].fHarmRatio = fHarmRatio;

    return 1;
}

/*! \brief  choose the best fundamental out of all the candidates
 *
 * \param pCHarmonic               array of candidates
 * \param iRefHarmonic             reference harmonic number
 * \param nGoodPeaks              number of candiates
 * \param fPrevFund                   reference fundamental
 * \return the integer number of the best candidate
 */
static int GetBestCandidate(SMS_HarmCandidate *pCHarmonic, 
                            int iRefHarmonic, int nGoodPeaks, sfloat fPrevFund)
{
    int iBestCandidate = 0, iPeak;
    sfloat fBestFreq, fHarmFreq, fDev;
  
    /* if a fundamental existed in previous frame take the closest candidate */
    if(fPrevFund > 0)
    {
        for(iPeak = 1; iPeak < nGoodPeaks; iPeak++)
        {
            if(fabs(fPrevFund - pCHarmonic[iPeak].fFreq / iRefHarmonic) <
               fabs(fPrevFund - pCHarmonic[iBestCandidate].fFreq / iRefHarmonic))
                iBestCandidate = iPeak;
        }
    }
    else
    {
        /* try to find the best candidate */
        for(iPeak = 1; iPeak < nGoodPeaks; iPeak++)
        {
            fBestFreq = pCHarmonic[iBestCandidate].fFreq / iRefHarmonic;
            fHarmFreq = fBestFreq * floor(.5 + 
                                          (pCHarmonic[iPeak].fFreq / iRefHarmonic) / 
                                           fBestFreq);
            fDev = fabs(fHarmFreq - (pCHarmonic[iPeak].fFreq / iRefHarmonic)) / fHarmFreq;
    
            /* if candidate is far from harmonic from best candidate and
             * bigger, take it */
            if(fDev > .2 &&
               pCHarmonic[iPeak].fMag > pCHarmonic[iBestCandidate].fMag)
                iBestCandidate = iPeak;
            /* if frequency deviation is much smaller, take it */
            else if(pCHarmonic[iPeak].fFreqDev < .2 * pCHarmonic[iBestCandidate].fFreqDev)
                iBestCandidate = iPeak;
            /* if freq. deviation is smaller and bigger amplitude, take it */
            else if(pCHarmonic[iPeak].fFreqDev < pCHarmonic[iBestCandidate].fFreqDev &&
                    pCHarmonic[iPeak].fMagPerc > pCHarmonic[iBestCandidate].fMagPerc &&
                    pCHarmonic[iPeak].fMag > pCHarmonic[iBestCandidate].fMag)
                iBestCandidate = iPeak;
        }
    }
    return iBestCandidate;
}

/*! \brief  main harmonic detection function
 *
 * find a given harmonic peak from a set of spectral peaks,     
 * put the frequency of the fundamental in the current frame
 *
 * \param pFrame                     pointer to current frame
 * \param fRefFundamental       frequency of previous frame
 * \param pPeakParams           pointer to analysis parameters
 * \todo is it possible to use pSpectralPeaks instead of SMS_AnalFrame?
 * \todo move pCHarmonic array to SMS_AnalFrame structure
  - this will allow for analysis of effectiveness from outside this file
 * This really should only be for sms_analyzeFrame
 */
sfloat sms_harmDetection(int numPeaks, SMS_Peak* spectralPeaks, sfloat refFundamental,
                         sfloat refHarmonic, sfloat lowestFreq, sfloat highestFreq,
                         int soundType, sfloat minRefHarmMag, sfloat refHarmMagDiffFromMax)
{
    int iPeak = -1, nGoodPeaks = 0, iCandidate, iBestCandidate;
    sfloat peakFreq=0;
    SMS_HarmCandidate pCHarmonic[N_HARM_PEAKS];

    /* find all possible candidates to use as harmonic reference */
    lowestFreq = lowestFreq * refHarmonic;
    highestFreq = highestFreq * refHarmonic;

    while((peakFreq < highestFreq) && (iPeak < numPeaks))
    {
        iPeak++;
        peakFreq = spectralPeaks[iPeak].fFreq;
        if(peakFreq > highestFreq)
            break;

        /* no more peaks */
        if(spectralPeaks[iPeak].fMag <= 0) /*!< \bug sfloat comparison to zero */
            break;

        /* peak too low */
        if(peakFreq < lowestFreq)
            continue;

        /* if previous fundamental look only around it */
        if(refFundamental > 0 &&
           fabs(peakFreq - (refHarmonic * refFundamental)) / refFundamental > .5)
            continue;

        iCandidate = GoodCandidate(iPeak, numPeaks, spectralPeaks, pCHarmonic,
                                   nGoodPeaks, soundType, refFundamental,
                                   minRefHarmMag, refHarmMagDiffFromMax, refHarmonic);

        /* good candiate found */
        if(iCandidate == 1)
            nGoodPeaks++;

        /* a perfect candiate found */
        else if(iCandidate == -2)
        {
            nGoodPeaks++;
            break;
        }
    }

    /* if no candidate for fundamental, continue */
    if(nGoodPeaks == 0)
        return -1;
    /* if only 1 candidate for fundamental take it */
    else if(nGoodPeaks == 1)
        return pCHarmonic[0].fFreq / refHarmonic;
    /* if more than one candidate choose the best one */
    else
    {
        iBestCandidate = GetBestCandidate(pCHarmonic, refHarmonic, nGoodPeaks, refFundamental);
        return pCHarmonic[iBestCandidate].fFreq / refHarmonic;
    }
}
