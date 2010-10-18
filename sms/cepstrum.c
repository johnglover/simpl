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
/*! \file cepstrum.c 
 * \brief routines for different Fast Fourier Transform Algorithms
 *
 */

#include "sms.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#define COEF ( 8 * powf(PI, 2)) 
#define CHOLESKY 1

typedef struct
{
        int nPoints;
        int nCoeff;
        gsl_matrix *pM;
        gsl_matrix *pMt;
        gsl_matrix *pR;
        gsl_matrix *pMtMR;
        gsl_vector *pXk;
        gsl_vector *pMtXk;
        gsl_vector *pC;
        gsl_permutation *pPerm;
} CepstrumMatrices;

void FreeDCepstrum(CepstrumMatrices *m)
{
        gsl_matrix_free(m->pM);
        gsl_matrix_free(m->pMt);
        gsl_matrix_free(m->pR);
        gsl_matrix_free(m->pMtMR);
        gsl_vector_free(m->pXk);
        gsl_vector_free(m->pMtXk);
        gsl_vector_free(m->pC);
        gsl_permutation_free (m->pPerm);

}

void AllocateDCepstrum(int nPoints, int nCoeff, CepstrumMatrices *m)
{
        if(m->nPoints != 0 || m->nCoeff != 0) 
                FreeDCepstrum(m);
        m->nPoints = nPoints;
        m->nCoeff = nCoeff;
        m->pM = gsl_matrix_alloc(nPoints, nCoeff);
        m->pMt = gsl_matrix_alloc(nCoeff, nPoints);
        m->pR = gsl_matrix_calloc(nCoeff, nCoeff);
        m->pMtMR = gsl_matrix_alloc(nCoeff, nCoeff);
        m->pXk = gsl_vector_alloc(nPoints);
        m->pMtXk = gsl_vector_alloc(nCoeff);
        m->pC = gsl_vector_alloc(nCoeff);
        m->pPerm = gsl_permutation_alloc (nCoeff);
}

/*! \brief Discrete Cepstrum Transform
 *
 * method for computing cepstrum aenalysis from a discrete
 * set of partial peaks (frequency and amplitude)
 *
 * This implementation is owed to the help of Jordi Janer (thanks!) from the MTG,
 * along with the following paper:
 * "Regularization Techniques for Discrete Cepstrum Estimation"
 * Olivier Cappe and Eric Moulines, IEEE Signal Processing Letters, Vol. 3
 * No.4, April 1996
 *
 * \todo add anchor point add at frequency = 0 with the same magnitude as the first
 * peak in pMag.  This does not change the size of the cepstrum, only helps to smoothen it
 * at the very beginning.
 *
 * \param sizeCepstrum order+1 of the discrete cepstrum
 * \param pCepstrum pointer to output array of cepstrum coefficients
 * \param sizeFreq number of partials peaks (the size of pFreq should be the same as pMag
 * \param pFreq pointer to partial peak frequencies (hertz)
 * \param pMag pointer to partial peak magnitudes (linear)
 * \param fLambda regularization factor
 * \param iMaxFreq maximum frequency of cepstrum
 */
void sms_dCepstrum( int sizeCepstrum, sfloat *pCepstrum, int sizeFreq, sfloat *pFreq, sfloat *pMag, 
                    sfloat fLambda, int iMaxFreq)
{
        int i, k;
        sfloat factor;
        sfloat fNorm = PI  / (float)iMaxFreq; /* value to normalize frequencies to 0:0.5 */
        //static sizeCepstrumStatic
        static CepstrumMatrices m;
        //printf("nPoints: %d, nCoeff: %d \n", m.nPoints, m.nCoeff);
        if(m.nPoints != sizeCepstrum || m.nCoeff != sizeFreq)
                AllocateDCepstrum(sizeFreq, sizeCepstrum, &m);
        int s; /* signum: "(-1)^n, where n is the number of interchanges in the permutation." */
        /* compute matrix M (eq. 4)*/
	for (i=0; i<sizeFreq; i++)
	{
                gsl_matrix_set (m.pM, i, 0, 1.); // first colum is all 1
		for (k=1; k <sizeCepstrum; k++)
                        gsl_matrix_set (m.pM, i, k , 2.*sms_sine(PI_2 + fNorm * k * pFreq[i]) );
	}

        /* compute transpose of M */
        gsl_matrix_transpose_memcpy (m.pMt, m.pM);
                               
        /* compute R diagonal matrix (for eq. 7)*/
        factor = COEF * (fLambda / (1.-fLambda)); /* \todo why is this divided like this again? */
	for (k=0; k<sizeCepstrum; k++)
                gsl_matrix_set(m.pR, k, k, factor * powf((sfloat) k,2.));

        /* MtM = Mt * M, later will add R */
        gsl_blas_dgemm  (CblasNoTrans, CblasNoTrans, 1., m.pMt, m.pM, 0.0, m.pMtMR);
        /* add R to make MtMR */
        gsl_matrix_add (m.pMtMR, m.pR);

        /* set pMag in X and multiply with Mt to get pMtXk */
        for(k = 0; k <sizeFreq; k++)
                gsl_vector_set(m.pXk, k, log(pMag[k]));
        gsl_blas_dgemv (CblasNoTrans, 1., m.pMt, m.pXk, 0., m.pMtXk);

        /* solve x (the cepstrum) in Ax = b, where A=MtMR and b=pMtXk */ 

        /* ==== the Cholesky Decomposition way ==== */
        /* MtM is 'symmetric and positive definite?' */
        //gsl_linalg_cholesky_decomp (m.pMtMR);
        //gsl_linalg_cholesky_solve (m.pMtMR, m.pMtXk, m.pC);

        /* ==== the LU decomposition way ==== */
        gsl_linalg_LU_decomp (m.pMtMR, m.pPerm, &s);
        gsl_linalg_LU_solve (m.pMtMR, m.pPerm, m.pMtXk, m.pC);

        
        /* copy pC to pCepstrum */
        for(i = 0; i  < sizeCepstrum; i++)
                pCepstrum[i] = gsl_vector_get (m.pC, i);
}

/*! \brief Spectrum Envelope from Cepstrum
 *
 *  from a set of cepstrum coefficients, compute the spectrum envelope
 *  
 * \param sizeCepstrum order + 1 of the cepstrum 
 * \param pCepstrum pointer to array of cepstrum coefficients
 * \param sizeEnv  size of spectrum envelope (max frequency in bins) \todo does this have to be a pow2
 * \param pEnv pointer to output spectrum envelope (real part only)
 */
void sms_dCepstrumEnvelope(int sizeCepstrum, sfloat *pCepstrum, int sizeEnv, sfloat *pEnv)
{
        
        static sfloat *pFftBuffer;
        static int sizeFftArray = 0;
        int sizeFft = sizeEnv << 1;
        int i;
        if(sizeFftArray != sizeFft)
        {
                if(sizeFftArray != 0) free(pFftBuffer);
                sizeFftArray = sms_power2(sizeFft);
                if(sizeFftArray != sizeFft)
                {
                        sms_error("bad fft size, incremented to power of 2");
                }
                if ((pFftBuffer = (sfloat *) malloc(sizeFftArray * sizeof(float))) == NULL)
                {
                        sms_error("could not allocate memory for fft array");
                        return;
                }
        }
        memset(pFftBuffer, 0, sizeFftArray * sizeof(sfloat));

        pFftBuffer[0] = pCepstrum[0] * 0.5;
        for (i = 1; i < sizeCepstrum-1; i++)
                pFftBuffer[i] = pCepstrum[i];


        sms_fft(sizeFftArray, pFftBuffer);

        for (i = 0; i < sizeEnv; i++)
                pEnv[i] = powf(EXP, 2. * pFftBuffer[i*2]);
}

/*! \brief main function for computing spectral envelope from sinusoidal peaks
 *
 * Magnitudes should already be in linear for this function.
 * If pSmsData->iEnvelope == SMS_ENV_CEP, will return cepstrum coefficeints
 * If pSmsData->iEnvelope == SMS_ENV_FBINS, will return linear magnitude spectrum
 * 
 * \param pSmsData pointer to SMS_Data structure with all the arrays necessary
 * \param pSpecEnvParams pointer to a structure of parameters for spectral enveloping
 */
void sms_spectralEnvelope( SMS_Data *pSmsData, SMS_SEnvParams *pSpecEnvParams)
{
        int i, k;
        int sizeCepstrum = pSpecEnvParams->iOrder+1;
        //int nPeaks = 0;
        static sfloat pFreqBuff[1000], pMagBuff[1000];
       
        /* \todo see if this memset is even necessary, once working */
        //memset(pSmsData->pSpecEnv, 0, pSpecEnvParams->nCoeff * sizeof(sfloat));

        /* try to store cepstrum coefficients in pSmsData->nEnvCoeff always.
           if cepstrum is what is wanted, memset the rest. otherwise, hand this array 2x to dCepstrumEnvelope */
        if(pSpecEnvParams->iOrder + 1> pSmsData->nEnvCoeff)
        {
                sms_error("cepstrum order is larger than the size of the spectral envelope");
                return;
        }

        /* find out how many tracks were actually found... many are zero
           \todo is this necessary? */
        for(i = 0, k=0; i < pSmsData->nTracks; i++)
        {
                if(pSmsData->pFSinFreq[i] > 0.00001)
                {
                        if(pSpecEnvParams->iAnchor != 0)
                        {
                                if(k == 0) /* add anchor at beginning */

                                {
                                        pFreqBuff[k] = 0.0;
                                        pMagBuff[k] = pSmsData->pFSinAmp[i];
                                        k++;
                                }
                        }
                        pFreqBuff[k] = pSmsData->pFSinFreq[i];
                        pMagBuff[k] = pSmsData->pFSinAmp[i];
                        k++;
                }
        }
        /* \todo see if adding an anchor at the max freq helps */
        

        if(k < 1) // how few can this be?  try out a few in python
                return;
        sms_dCepstrum(sizeCepstrum, pSmsData->pSpecEnv, k, pFreqBuff, pMagBuff, 
                      pSpecEnvParams->fLambda, pSpecEnvParams->iMaxFreq);

        if(pSpecEnvParams->iType == SMS_ENV_FBINS)
        {
                sms_dCepstrumEnvelope(sizeCepstrum, pSmsData->pSpecEnv, 
                                      pSpecEnvParams->nCoeff, pSmsData->pSpecEnv);
        }
}
