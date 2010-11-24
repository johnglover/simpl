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
/*! \file peakDetection.c
 * \brief peak detection algorithm and functions
 */

#include "sms.h"

/*! \brief function used for the parabolic interpolation of the spectral peaks
 *
 * it performs the interpolation in a log scale and
 * stores the location in pFDiffFromMax and
 *
 * \param fMaxVal        value of max bin 
 * \param fLeftBinVal    value for left bin
 * \param fRightBinVal   value for right bin
 * \param pFDiffFromMax location of the tip as the difference from the top bin 
 *  \return the peak height 
 */
static sfloat PeakInterpolation (sfloat fMaxVal, sfloat fLeftBinVal, 
                                sfloat fRightBinVal, sfloat *pFDiffFromMax)
{
    /* get the location of the tip of the parabola */
    *pFDiffFromMax = (.5 * (fLeftBinVal - fRightBinVal) /
        (fLeftBinVal - (2*fMaxVal) + fRightBinVal));
    /* return the value at the tip */
    return(fMaxVal - (.25 * (fLeftBinVal - fRightBinVal) * 
                      *pFDiffFromMax));
}

/*! \brief detect the next local maximum in the spectrum
 *
 * stores the value in pFMaxVal 
 *                                      
 * \todo export this to sms.h and wrap in pysms
 *
 * \param pFMagSpectrum   magnitude spectrum 
 * \param iHighBinBound      highest bin to search
 * \param pICurrentLoc      current bin location
 * \param pFMaxVal        value of the maximum found
 * \param fMinPeakMag  minimum magnitude to accept a peak
 * \return the bin location of the maximum  
 */
static int FindNextMax ( sfloat *pFMagSpectrum, int iHighBinBound, 
                        int *pICurrentLoc, sfloat *pFMaxVal, sfloat fMinPeakMag)
{
    int iCurrentBin = *pICurrentLoc;
    sfloat fPrevVal = pFMagSpectrum[iCurrentBin - 1];
        sfloat fCurrentVal = pFMagSpectrum[iCurrentBin];
        sfloat fNextVal = (iCurrentBin >= iHighBinBound) 
                ? 0 : pFMagSpectrum[iCurrentBin + 1];
  
    /* try to find a local maximum */
    while (iCurrentBin <= iHighBinBound)
    {
        if (fCurrentVal > fMinPeakMag &&
           fCurrentVal >= fPrevVal &&
           fCurrentVal >= fNextVal)
            break;
        iCurrentBin++;
        fPrevVal = fCurrentVal;
        fCurrentVal = fNextVal;
        fNextVal = pFMagSpectrum[1+iCurrentBin];
    }
    /* save the current location, value of maximum and return */
    /*    location of max */
    *pICurrentLoc = iCurrentBin + 1;
    *pFMaxVal = fCurrentVal;
    return(iCurrentBin);
}

/*! \brief function to detect the next spectral peak 
 *                           
 * \param pFMagSpectrum    magnitude spectrum 
 * \param iHighestBin       highest bin to search
 * \param pICurrentLoc       current bin location
 * \param pFPeakMag        magnitude value of peak
 * \param pFPeakLoc        location of peak
 * \param fMinPeakMag  minimum magnitude to accept a peak
 * \return 1 if found, 0 if not  
 */
static int FindNextPeak (sfloat *pFMagSpectrum, int iHighestBin, 
                         int *pICurrentLoc, sfloat *pFPeakMag, sfloat *pFPeakLoc,
                         sfloat fMinPeakMag)
{
    int iPeakBin = 0;       /* location of the local peak */
    sfloat fPeakMag = 0;         /* value of local peak */
  
    /* keep trying to find a good peak while inside the freq range */
    while ((iPeakBin = FindNextMax(pFMagSpectrum, iHighestBin, 
           pICurrentLoc, &fPeakMag, fMinPeakMag)) 
           <= iHighestBin)
    {
        /* get the neighboring samples */
        sfloat fDiffFromMax = 0;
        sfloat fLeftBinVal = pFMagSpectrum[iPeakBin - 1];
        sfloat fRightBinVal = pFMagSpectrum[iPeakBin + 1];
        if (fLeftBinVal <= 0 || fRightBinVal <= 0) //ahah! there you are!
            continue;
        /* interpolate the spectral samples to obtain
           a more accurate magnitude and freq */
        *pFPeakMag = PeakInterpolation (fPeakMag, fLeftBinVal,
                                        fRightBinVal, &fDiffFromMax);
        *pFPeakLoc = iPeakBin + fDiffFromMax;
        return (1);
    }
    /* if it does not find a peak return 0 */
    return (0);
}

/*! \brief get the corresponding phase value for a given peak
 *
 * performs linear interpolation for a more accurate phase

 * \param pPhaseSpectrum     phase spectrum
 * \param fPeakLoc                 location of peak
 * \return the phase value                             
 */
static sfloat GetPhaseVal (sfloat *pPhaseSpectrum, sfloat fPeakLoc)
{
    int bin = (int) fPeakLoc;
    sfloat fFraction = fPeakLoc - bin,
        fLeftPha = pPhaseSpectrum[bin],
        fRightPha = pPhaseSpectrum[bin+1];
  
    /* check for phase wrapping */
    if (fLeftPha - fRightPha > 1.5 * PI)
        fRightPha += TWO_PI;
    else if (fRightPha - fLeftPha > 1.5 * PI)
        fLeftPha += TWO_PI;
  
    /* return interpolated phase */
    return (fLeftPha + fFraction * (fRightPha - fLeftPha));
}

/*! \brief find the prominent spectral peaks
 * 
 * uses a dB spectrum
 *
 * \param sizeSpec size of magnitude spectrum
 * \param pMag pointer to power spectrum
 * \param pPhase pointer to phase spectrum
 * \param pSpectralPeaks pointer to array of peaks
 * \param pAnalParams peak detection parameters
 * \return the number of peaks found
 */
int sms_detectPeaks(int sizeSpec, sfloat *pMag, sfloat *pPhase,
                    SMS_Peak *pSpectralPeaks, SMS_AnalParams *pAnalParams)
{
    int sizeFft = sizeSpec << 1;
    sfloat fInvSizeFft = 1.0 / sizeFft;
    int iFirstBin = MAX(1, sizeFft * pAnalParams->fLowestFreq / pAnalParams->iSamplingRate);
    int iHighestBin = MIN(sizeSpec-1, sizeFft * pAnalParams->fHighestFreq / pAnalParams->iSamplingRate);

    /* clear peak structure */
    memset(pSpectralPeaks, 0, pAnalParams->maxPeaks * sizeof(SMS_Peak));

    /* set starting search values */
    int iCurrentLoc = iFirstBin;
    int iPeak = 0;          /* index for spectral search */
    sfloat fPeakMag = 0.0;  /* magnitude of peak */
    sfloat fPeakLoc = 0.0;  /* location of peak */

    /* find peaks */
    while((iPeak < pAnalParams->maxPeaks) &&
          (FindNextPeak(pMag, iHighestBin, &iCurrentLoc, &fPeakMag, 
                        &fPeakLoc, pAnalParams->fMinPeakMag) == 1))
    {
        /* store peak values */
        pSpectralPeaks[iPeak].fFreq = pAnalParams->iSamplingRate * fPeakLoc * fInvSizeFft;
        pSpectralPeaks[iPeak].fMag = fPeakMag;
        pSpectralPeaks[iPeak].fPhase = GetPhaseVal(pPhase, fPeakLoc);
        iPeak++;
    }

    /* return the number of peaks found */
    return iPeak;
}
