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
/*! \file filters.c
 * \brief various filters
 */

#include "sms.h"

/*! \brief coefficient for pre_emphasis filter */
#define SMS_EMPH_COEF    .9   

/* pre-emphasis filter function, it returns the filtered value   
 *
 * sfloat fInput;   sound sample
 */
sfloat sms_preEmphasis(sfloat fInput, SMS_AnalParams *pAnalParams)
{
    sfloat fOutput = fInput - SMS_EMPH_COEF * pAnalParams->preEmphasisLastValue;
    pAnalParams->preEmphasisLastValue = fOutput;
    return fOutput;
}

/* de-emphasis filter function, it returns the filtered value 
 *
 * sfloat fInput;   sound input
 */
sfloat sms_deEmphasis(sfloat fInput, SMS_SynthParams *pSynthParams)
{
    sfloat fOutput = fInput + SMS_EMPH_COEF * pSynthParams->deEmphasisLastValue;
    pSynthParams->deEmphasisLastValue = fInput;
    return fOutput;
}

/*! \brief  function to implement a zero-pole filter
 * 
 * \todo will forgetting to reset  pD to zero at the beginning of a new analysis
 * (when there are multiple analyses within the life of one program)
 * cause problems?
 *
 * \param pFa        pointer to numerator coefficients
 * \param pFb        pointer to denominator coefficients
 * \param nCoeff    number of coefficients
 * \param fInput     input sample
 * \return value is the  filtered sample 
 */
static sfloat ZeroPoleFilter(sfloat *pFa, sfloat *pFb, int nCoeff, sfloat fInput )
{
	double fOut = 0;
	int iSection;
    static sfloat pD[5] = {0, 0, 0, 0, 0};

	pD[0] = fInput;
	for (iSection = nCoeff-1; iSection > 0; iSection--)
	{
		fOut = fOut + pFa[iSection] * pD[iSection];
		pD[0] = pD[0] - pFb[iSection] * pD[iSection];
		pD[iSection] = pD [iSection-1];
	}
	fOut = fOut + pFa[0] * pD[0];
	return (sfloat) fOut;
}

/*! \brief function to filter a waveform with a high-pass filter
 * 
 *  cutoff =1500 Hz  
 * 
 * \todo this filter only works on sample rates up to 48k?
 *
 * \param sizeResidual        size of signal
 * \param pResidual          pointer to residual signal
 * \param iSamplingRate      sampling rate of signal                                                    
 */
void sms_filterHighPass(int sizeResidual, sfloat *pResidual, int iSamplingRate)
{
	/* cutoff 800Hz */
	static sfloat pFCoeff32k[10] =  {0.814255, -3.25702, 4.88553, -3.25702, 
		0.814255, 1, -3.58973, 4.85128, -2.92405, 0.66301};
	static sfloat pFCoeff36k[10] =  {0.833098, -3.33239, 4.99859, -3.33239, 
		0.833098, 1, -3.63528, 4.97089, -3.02934,0.694052};
	static sfloat pFCoeff40k[10] =  {0.848475, -3.3939, 5.09085, -3.3939, 
		0.848475, 1, -3.67173, 5.068, -3.11597, 0.71991}; 
	static sfloat pFCoeff441k[10] =  {0.861554, -3.44622, 5.16932, -3.44622, 
		0.861554, 1, -3.70223, 5.15023, -3.19013, 0.742275};
	static sfloat pFCoeff48k[10] =  {0.872061, -3.48824, 5.23236, -3.48824, 
		0.872061, 1, -3.72641, 5.21605, -3.25002, 0.76049};
	sfloat *pFCoeff, fSample = 0;
	int i;
  
	if(iSamplingRate <= 32000)
		pFCoeff = pFCoeff32k;
	else if(iSamplingRate <= 36000)
		pFCoeff = pFCoeff36k;
	else if(iSamplingRate <= 40000)
		pFCoeff = pFCoeff40k;
	else if(iSamplingRate <= 44100)
		pFCoeff = pFCoeff441k;
	else
		pFCoeff = pFCoeff48k;
  
	for(i = 0; i < sizeResidual; i++)
	{
		/* try to avoid underflow when there is nothing to filter */
		if(i > 0 && fSample == 0 && pResidual[i] == 0)
			return;
      
		fSample = pResidual[i];
		pResidual[i] = ZeroPoleFilter (&pFCoeff[0], &pFCoeff[5], 5, fSample);
	}
}

/*! \brief a spectral filter
 *
 * filter each point of the current array by the surounding
 * points using a triangular window
 *
 * \param pFArray	        two dimensional input array
 * \param size1		vertical size of pFArray
 * \param size2		horizontal size of pFArray
 * \param pFOutArray     output array of size size1
 */
void sms_filterArray(sfloat *pFArray, int size1, int size2, sfloat *pFOutArray)
{
	int i, j, iPoint, iFrame, size2_2 = size2-2, size2_1 = size2-1;
	sfloat *pFCurrentArray = pFArray + (size2_1) * size1;
    sfloat fVal, fWeighting, fTotalWeighting, fTmpVal;

	/* find the filtered envelope */
	for(i = 0; i < size1; i++)
	{
		fVal = pFCurrentArray[i];
		fTotalWeighting = 1;
		/* filter by the surrounding points */
		for(j = 1; j < (size2_2); j++)
		{
			fWeighting = (sfloat) size2 / (1+ j);
			/* filter on the vertical dimension */
			/* consider the lower points */
			iPoint = i - (size2_1) + j;
			if(iPoint >= 0)
			{  
				fVal += pFCurrentArray[iPoint] * fWeighting;
				fTotalWeighting += fWeighting;
			}
			/* consider the higher points */
			iPoint = i + (size2_1) - j;
			if(iPoint < size1)
			{
				fVal += pFCurrentArray[iPoint] * fWeighting;
				fTotalWeighting += fWeighting;
			}
			/*filter on the horizontal dimension */
			/* consider the previous points */
 			iFrame = j;
			fTmpVal = pFArray[iFrame*size1 + i];
			if(fTmpVal)
			{
				fVal += fTmpVal * fWeighting;
				fTotalWeighting += fWeighting;
			}
		}
		/* scale value by weighting */
		pFOutArray[i] = fVal / fTotalWeighting;
	}
}
