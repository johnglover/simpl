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
/*! \file soundIO.c
 * \brief soundfile input and output.
*/
#include "sms.h"

/*! \brief fill the sound buffer
 *
 * \param sizeWaveform  size of input data
 * \param pWaveform     input data
 * \param pAnalParams   pointer to structure of analysis parameters
 */
void sms_fillSoundBuffer(int sizeWaveform, sfloat *pWaveform, SMS_AnalParams *pAnalParams)
{
    int i;
    long sizeNewData = (long)sizeWaveform;

    /* leave space for new data */
    memcpy(pAnalParams->soundBuffer.pFBuffer, pAnalParams->soundBuffer.pFBuffer+sizeNewData,
           sizeof(sfloat) * (pAnalParams->soundBuffer.sizeBuffer - sizeNewData));

    pAnalParams->soundBuffer.iFirstGood = MAX(0, pAnalParams->soundBuffer.iFirstGood - sizeNewData);
    pAnalParams->soundBuffer.iMarker += sizeNewData;

    /* put the new data in, and do some pre-emphasis */
    if(pAnalParams->iAnalysisDirection == SMS_DIR_REV)
        for(i=0; i<sizeNewData; i++)
            pAnalParams->soundBuffer.pFBuffer[pAnalParams->soundBuffer.sizeBuffer - sizeNewData + i] =
                sms_preEmphasis(pWaveform[sizeNewData - (1 + i)], pAnalParams);
    else
        for(i=0; i<sizeNewData; i++)
            pAnalParams->soundBuffer.pFBuffer[pAnalParams->soundBuffer.sizeBuffer - sizeNewData + i] =
                sms_preEmphasis(pWaveform[i], pAnalParams);
}

