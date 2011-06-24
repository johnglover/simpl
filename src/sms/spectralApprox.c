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
/*! \file spectralApprox.c
 * \brief line segment approximation of a magnitude spectrum
 */
#include "sms.h"

/*! \brief approximate a magnitude spectrum
 * First downsampling using local maxima and then upsampling using linear 
 * interpolation. The output spectrum doesn't have to be the same size as 
 * the input one.
 *
 * \param pFSpec1       magnitude spectrum to approximate
 * \param sizeSpec1     size of input spectrum
 * \param sizeSpec1Used size of the spectrum to use
 * \param pFSpec2       output envelope
 * \param sizeSpec2     size of output envelope
 * \param nCoefficients number of coefficients to use in approximation
 * \return error code \see SMS_ERRORS
 */
int sms_spectralApprox(sfloat *pFSpec1, int sizeSpec1, int sizeSpec1Used,
                       sfloat *pFSpec2, int sizeSpec2, int nCoefficients,
                       sfloat *envelope)
{
    sfloat fHopSize, fCurrentLoc = 0, fLeft = 0, fRight = 0, fValue = 0, 
           fLastLocation, fSizeX, fSpec2Acum=0, fNextHop, fDeltaY;
    int iFirstGood = 0, iLastSample = 0, i, j;

    /* when number of coefficients is smaller than 2 do not approximate */
    if(nCoefficients < 2)
    {
        for(i = 0; i < sizeSpec2; i++)
            pFSpec2[i] = 1;
        return SMS_OK;
    }

    /* calculate the hop size */
    if(nCoefficients > sizeSpec1)
        nCoefficients = sizeSpec1;

    fHopSize = (sfloat)sizeSpec1Used / nCoefficients;

    /* approximate by linear interpolation */
    if(fHopSize > 1)
    {
        iFirstGood = 0;
        for(i = 0; i < nCoefficients; i++)
        {
            iLastSample = fLastLocation = fCurrentLoc + fHopSize;
            iLastSample = MIN(sizeSpec1-1, iLastSample);
            if(iLastSample < sizeSpec1-1)
            {
                fRight = pFSpec1[iLastSample] +
                         (pFSpec1[iLastSample+1] - pFSpec1[iLastSample]) * 
                         (fLastLocation - iLastSample);
            }
            else
            {
                fRight = pFSpec1[iLastSample];
            }

            fValue = 0;
            for(j = iFirstGood; j <= iLastSample; j++)
                fValue = MAX (fValue, pFSpec1[j]);
            fValue = MAX(fValue, MAX (fRight, fLeft));
            envelope[i] = fValue;

            fLeft = fRight;
            fCurrentLoc = fLastLocation;
            iFirstGood = (int)(1+ fCurrentLoc);
        }
    }
    else if(fHopSize == 1)
    {
        for(i = 0; i < nCoefficients; i++)
            envelope[i] = pFSpec1[i];
    }
    else
    {
        sms_error("SpectralApprox: sizeSpec1 has too many nCoefficients"); /* \todo need to increase the frequency? */
        return -1;
    }

    /* Creates Spec2 from Envelope */
    if(nCoefficients < sizeSpec2)
    {
        fSizeX = (sfloat) (sizeSpec2-1) / nCoefficients;

        /* the first step */
        fNextHop = fSizeX / 2;
        fDeltaY = envelope[0] / fNextHop;
        fSpec2Acum=pFSpec2[j=0]=0;
        while(++j < fNextHop)  
            pFSpec2[j] = (fSpec2Acum += fDeltaY);

        /* middle values */
        for(i = 0; i <= nCoefficients-2; ++i) 
        {
            fDeltaY = (envelope[i+1] - envelope[i]) / fSizeX;
            /* first point of a segment */
            pFSpec2[j] = (fSpec2Acum = (envelope[i]+(fDeltaY*(j-fNextHop))));
            ++j;
            /* remaining points */
            fNextHop += fSizeX;
            while(j < fNextHop)  
                pFSpec2[j++] = (fSpec2Acum += fDeltaY);
        }

        /* last step */
        fDeltaY = -envelope[i] * 2 / fSizeX;
        /* first point of the last segment */
        pFSpec2[j] = (fSpec2Acum = (envelope[i]+(fDeltaY*(j-fNextHop))));
        ++j;
        fNextHop += fSizeX / 2;
        while(j < sizeSpec2-1)  
            pFSpec2[j++]=(fSpec2Acum += fDeltaY);
        /* last should be exactly zero */
        pFSpec2[sizeSpec2-1] = .0;  
    }
    else if(nCoefficients == sizeSpec2)
    {
        for(i = 0; i < nCoefficients; i++)
            pFSpec2[i] = envelope[i];
    }
    else
    {
        sms_error("SpectralApprox: sizeSpec2 has too many nCoefficients\n");
        return -1;
    }

    return SMS_OK;
}


