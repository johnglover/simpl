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
/*! \file tables.c
 * \brief sin and sinc tables.
 * 
 * contains functions for creating and indexing the tables
 */
#include "sms.h"

/*! \brief value to scale the sine-table-lookup phase */
static sfloat fSineScale;
/*! \brief inverse of fSineScale - turns a division into multiplication */
static sfloat fSineIncr;
/*! \brief value to scale the sinc-table-lookup phase */
static sfloat fSincScale;
/*! \brief global pointer to the sine table */
static sfloat *sms_tab_sine;
/*! \brief global pointer to the sinc table */
static sfloat *sms_tab_sinc;

/*! \brief prepares the sine table
 * \param  nTableSize    size of table
 * \return error code \see SMS_MALLOC in SMS_ERRORS
 */
int sms_prepSine(int nTableSize)
{
    register int i;
    sfloat fTheta;

    sms_tab_sine = (sfloat *)malloc(nTableSize * sizeof(sfloat));
    if(sms_tab_sine == NULL)
    {
        sms_error("Could not allocate memory for sine table");
        return SMS_MALLOC;
    }
    fSineScale =  (sfloat)(TWO_PI) / (sfloat)(nTableSize - 1);
    fSineIncr = 1.0 / fSineScale;
    fTheta = 0.0;
    for(i = 0; i < nTableSize; i++) 
    {
        fTheta = fSineScale * (sfloat)i;
        sms_tab_sine[i] = sin(fTheta);
    }
    return SMS_OK;
}
/*! \brief clear sine table */
void sms_clearSine()
{
    if(sms_tab_sine)
        free(sms_tab_sine);
    sms_tab_sine = NULL;
}

/*! \brief table-lookup sine method
 * \param fTheta    angle in radians
 * \return approximately sin(fTheta)
 */
sfloat sms_sine(sfloat fTheta)
{
    int i;
    fTheta = fTheta - floor(fTheta * INV_TWO_PI) * TWO_PI;

    if(fTheta < 0)
    {
        i =  .5 - (fTheta * fSineIncr);
        return -(sms_tab_sine[i]);
    }
    else
    {
        i = fTheta * fSineIncr + .5;
        return sms_tab_sine[i];
    }
}

/*! \brief Sinc method to generate the lookup table
 */
static sfloat Sinc(sfloat x, sfloat N)	
{
	return sinf((N/2) * x) / sinf(x/2);
}

/*! \brief prepare the Sinc table
 *
 * used for the main lobe of a frequency domain 
 * BlackmanHarris92 window
 *
 * \param  nTableSize    size of table
 * \return error code \see SMS_MALLOC in SMS_ERRORS
 */
int sms_prepSinc(int nTableSize)
{
    int i, m;
	sfloat N = 512.0;
	sfloat fA[4] = {.35875, .48829, .14128, .01168};
    sfloat fMax = 0;
    sfloat fTheta = -4.0 * TWO_PI / N;
    sfloat fThetaIncr = (8.0 * TWO_PI / N) / (nTableSize);

    sms_tab_sinc = (sfloat *)calloc(nTableSize, sizeof(sfloat));
    if(sms_tab_sinc == NULL)
    {
        sms_error("Could not allocate memory for sinc table");
        return (SMS_MALLOC);
    }
    
    for(i = 0; i < nTableSize; i++) 
    {
        for (m = 0; m < 4; m++)
            sms_tab_sinc[i] +=  -1 * (fA[m]/2) * 
                (Sinc (fTheta - m * TWO_PI/N, N) + 
                 Sinc (fTheta + m * TWO_PI/N, N));
        fTheta += fThetaIncr;
    }

    fMax = sms_tab_sinc[(int) nTableSize / 2];
    for (i = 0; i < nTableSize; i++) 
        sms_tab_sinc[i] = sms_tab_sinc[i] / fMax;
    
    fSincScale = (sfloat) nTableSize / 8.0;
    return SMS_OK;
}

/*! \brief clear sine table */
void sms_clearSinc()
{
    if(sms_tab_sinc)
        free(sms_tab_sinc);
    sms_tab_sinc = 0;
}

/*! \brief global sinc table-lookup method
 * 
 * fTheta has to be from 0 to 8
 *
 * \param fTheta    angle in radians
 * \return approximately sinc(fTheta)
 */
sfloat sms_sinc(sfloat fTheta)
{
	int index = (int) (.5 + fSincScale * fTheta);
	return sms_tab_sinc[index];
}


