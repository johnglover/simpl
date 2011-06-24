/* 
 * Copyright (c) 2009 John Glover, National University of Ireland, Maynooth
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

/*! \file modify.c
 * \brief modify sms data 
 */

#include "sms.h"

/*! \brief initialize a modifications structure based on an SMS_Header
 *
 * \param params pointer to parameter structure
 * \param header pointer to sms header
 */
void sms_initModify(SMS_Header *header, SMS_ModifyParams *params)
{
        static int sizeEnvArray = 0;
        params->maxFreq = header->iMaxFreq;
        params->sizeSinEnv = header->nEnvCoeff;

        if(sizeEnvArray < params->sizeSinEnv)
        {
                if(sizeEnvArray != 0) free(params->sinEnv);
                if ((params->sinEnv = (sfloat *) malloc(params->sizeSinEnv * sizeof(sfloat))) == NULL)
                {
                        sms_error("could not allocate memory for envelope array");
                        return;
                }
                sizeEnvArray = params->sizeSinEnv;
        }
        params->ready = 1;
}

/*! \brief initialize modification parameters
 *
 * \todo call this from sms_initSynth()? some other mod params are updated there
 *
 * \param params pointer to parameters structure
 */
void sms_initModifyParams(SMS_ModifyParams *params)
{
        params->ready = 0;
	params->doResGain = 0;
	params->resGain = 1.;
	params->doTranspose = 0;
	params->transpose = 0;
	params->doSinEnv = 0;
	params->sinEnvInterp = 0.;
	params->sizeSinEnv = 0;
	params->doResEnv = 0;
	params->resEnvInterp = 0.;
	params->sizeResEnv = 0;
}

/*! \brief free memory allocated during initialization
 *
 * \param params pointer to parameter structure
 */
void sms_freeModify(SMS_ModifyParams *params)
{
}

/*! \brief linear interpolation between 2 spectral envelopes.
 *
 * The values in env2 are overwritten by the new interpolated envelope values.
 */
void sms_interpEnvelopes(int sizeEnv, sfloat *env1, sfloat *env2, sfloat interpFactor)
{
        if(sizeEnv <= 0)
        {       
                return;
        }
        
        int i;
        sfloat amp1, amp2;
        
        for(i = 0; i < sizeEnv; i++)
        {
                amp1 = env1[i];
                amp2 = env2[i];
                if(amp1 <= 0) amp1 = amp2;
                if(amp2 <= 0) amp2 = amp1;
                env2[i] = amp1 + (interpFactor * (amp2 - amp1));
        }
}

/*! \brief apply the spectral envelope of 1 sound to another 
 *
 * Changes the amplitude of spectral peaks in a target sound (pFreqs, pMags) to match those
 * in the envelope (pCepEnvFreqs, pCepEnvMags) of another, up to a maximum frequency of maxFreq.
 */
void sms_applyEnvelope(int numPeaks, sfloat *pFreqs, sfloat *pMags, int sizeEnv, sfloat *pEnvMags, int maxFreq)
{
	if(sizeEnv <= 0 || maxFreq <= 0)
        {
                return;
        }

	int i, envPos;
    sfloat frac, binSize = (sfloat)maxFreq / (sfloat)sizeEnv;
        
        for(i = 0; i < numPeaks; i++)
	{
                /* convert peak freqs into bin positions for quicker envelope lookup */
                /* \todo try to remove so many pFreq lookups and get rid of divide */
                pFreqs[i] /= binSize;

		/* if current peak is within envelope range, set its mag to the envelope mag */
		if(pFreqs[i] < (sizeEnv-1) && pFreqs[i] > 0)
		{
                        envPos = (int)pFreqs[i];
                        frac = pFreqs[i] - envPos;
                        if(envPos < sizeEnv - 1)
                        {
			        pMags[i] = ((1.0 - frac) * pEnvMags[envPos]) + (frac * pEnvMags[envPos+1]);
                        }
                        else
                        {       
                                pMags[i] = pEnvMags[sizeEnv-1];
                        }
                }
		else
		{
			pMags[i] = 0;
		}
        
                /* convert back to frequency values */
                pFreqs[i] *= binSize;
	}

}

/*! \brief scale the residual gain factor
 * 
 * \param frame pointer to sms data
 * \param gain factor to scale the residual
 */
void sms_resGain(SMS_Data *frame, sfloat gain)
{
        int i;
        for( i = 0; i < frame->nCoeff; i++)
                frame->pFStocCoeff[i] *= gain;
}


/*! \brief basic transposition
 * Multiply the frequencies of the deterministic component by a constant
 */
void sms_transpose(SMS_Data *frame, sfloat transpositionFactor)
{
        int i;
        for(i = 0; i < frame->nTracks; i++)
        {
                frame->pFSinFreq[i] *= sms_scalarTempered(transpositionFactor);
        }
}


/*! \brief transposition maintaining spectral envelope
 *
 * Multiply the frequencies of the deterministic component by a constant, then change
 * their amplitudes so that the original spectral envelope is maintained
 */
void sms_transposeKeepEnv(SMS_Data *frame, sfloat transpositionFactor, int maxFreq)
{
	sms_transpose(frame, transpositionFactor);
	sms_applyEnvelope(frame->nTracks, frame->pFSinFreq, frame->pFSinAmp, frame->nEnvCoeff, frame->pSpecEnv, maxFreq);
}

/*! \brief modify a frame (SMS_Data object) 
 *
 * Performs a modification on a SMS_Data object. The type of modification and any additional
 * parameters are specified in the given SMS_ModifyParams structure.
 */
void sms_modify(SMS_Data *frame, SMS_ModifyParams *params)
{
	if(params->doResGain)
                sms_resGain(frame, params->resGain);

	if(params->doTranspose)
                sms_transpose(frame, params->transpose);
	
	if(params->doSinEnv)
	{
		if(params->sinEnvInterp < .00001) /* maintain original */
			sms_applyEnvelope(frame->nTracks, frame->pFSinFreq, frame->pFSinAmp,
					  frame->nEnvCoeff, frame->pSpecEnv, params->maxFreq);
		else
		{
		    if(params->sinEnvInterp > .00001 && params->sinEnvInterp < .99999)
			    sms_interpEnvelopes(params->sizeSinEnv, frame->pSpecEnv, params->sinEnv, params->sinEnvInterp);
		   
		    sms_applyEnvelope(frame->nTracks, frame->pFSinFreq, frame->pFSinAmp,
				      params->sizeSinEnv, params->sinEnv, params->maxFreq);

		}
	}
}

