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
/*! \file sms.c
 * \brief initialization, free, and debug functions
 */

#include "sms.h"
#include "SFMT.h" /*!< mersenne twister random number genorator */

char *pChDebugFile = "debug.txt"; /*!< debug text file */
FILE *pDebug; /*!< pointer to debug file */

static char error_message[256];
static int error_status = 0;
static sfloat mag_thresh = .00001; /*!< magnitude threshold for db conversion (-100db)*/
static sfloat inv_mag_thresh = 100000.; /*!< inv(.00001) */
static int initIsDone = 0; /* \todo is this variable necessary? */ 

#define SIZE_TABLES 4096
#define HALF_MAX 1073741823.5  /*!< half the max of a 32-bit word */
#define INV_HALF_MAX (1.0 / HALF_MAX)
#define TWENTY_OVER_LOG10 (20. / LOG10)

/*! \brief initialize global data
 *
 * Currently, just generating the sine and sinc tables.
 * This is necessary before both analysis and synthesis.
 *
 * If using the Mersenne Twister algorithm for random number
 * generation, initialize (seed) it.
 *
 * \return error code \see SMS_MALLOC or SMS_OK in SMS_ERRORS
 */
int sms_init(void) 
{
    if (!initIsDone)
    {
        initIsDone = 1;
        if(sms_prepSine(SIZE_TABLES))
        {
            sms_error("cannot allocate memory for sine table");
            return -1;
        }
        if(sms_prepSinc(SIZE_TABLES))
        {
            sms_error("cannot allocate memory for sinc table");
            return -1;
        }

#ifdef MERSENNE_TWISTER
        init_gen_rand(1234);
#endif
    }

    return 0;
}

/*! \brief free global data
 *
 * deallocates memory allocated to global arrays (windows and tables)
 */
void sms_free()
{
    initIsDone = 0;
    sms_clearSine();
    sms_clearSinc();
}

/*! \brief give default values to an SMS_AnalParams struct 
 * 
 * This will initialize an SMS_AnalParams with values that work
 * for common analyses.  It is useful to start with and then
 * adjust the parameters manually to fit a particular sound
 *
 * Certain things are hard coded in here that will have to 
 * be updated later (i.e. samplerate), so it is best to call this
 * function first, then fill whatever parameters need to be 
 * adjusted.
 * 
 * \param pAnalParams    pointer to analysis data structure
 */
void sms_initAnalParams(SMS_AnalParams *pAnalParams)
{
    int i;
    pAnalParams->iDebugMode = 0;
    pAnalParams->iFormat = SMS_FORMAT_H;
    pAnalParams->iSoundType = SMS_SOUND_TYPE_MELODY;
    pAnalParams->iStochasticType =SMS_STOC_APPROX;
    pAnalParams->iFrameRate = 300;
    pAnalParams->nStochasticCoeff = 128;
    pAnalParams->fLowestFundamental = 50;
    pAnalParams->fHighestFundamental = 1000;
    pAnalParams->fDefaultFundamental = 100;
    pAnalParams->fPeakContToGuide = .4;
    pAnalParams->fFundContToGuide = .5;
    pAnalParams->fFreqDeviation = .45;
    pAnalParams->iSamplingRate = 44100; /* should be set to the real samplingrate with sms_initAnalysis */
    pAnalParams->iDefaultSizeWindow = 1001;
    pAnalParams->windowSize = 0;
    pAnalParams->sizeHop = 110;
    pAnalParams->fSizeWindow = 3.5;
    pAnalParams->nTracks = 60;
    pAnalParams->maxPeaks = 60;
    pAnalParams->nGuides = 100;
    pAnalParams->iCleanTracks = 1;
    pAnalParams->fMinRefHarmMag = 30;
    pAnalParams->fRefHarmMagDiffFromMax = 30;
    pAnalParams->iRefHarmonic = 1;
    pAnalParams->iMinTrackLength = 40; /*!< depends on iFrameRate normally */
    pAnalParams->iMaxSleepingTime = 40; /*!< depends on iFrameRate normally */
    pAnalParams->fLowestFreq = 50.0;
    pAnalParams->fHighestFreq = 12000.;
    pAnalParams->fMinPeakMag = 0.;
    pAnalParams->iAnalysisDirection = SMS_DIR_FWD;
    pAnalParams->iWindowType = SMS_WIN_BH_70;
    pAnalParams->iSizeSound = 0; /*!< no sound yet */
    pAnalParams->nFrames = 0; /*!< no frames yet */
    pAnalParams->minGoodFrames = 3;
    pAnalParams->maxDeviation = 0.01;
    pAnalParams->analDelay = 100;
    pAnalParams->iMaxDelayFrames = MAX(pAnalParams->iMinTrackLength, pAnalParams->iMaxSleepingTime) + 2 +
        (pAnalParams->minGoodFrames + pAnalParams->analDelay);
    pAnalParams->fResidualAccumPerc = 0.;
    pAnalParams->preEmphasis = 1; /*!< perform pre-emphasis by default */
    pAnalParams->preEmphasisLastValue = 0.;
    /* spectral envelope params */
    pAnalParams->specEnvParams.iType = SMS_ENV_NONE; /* turn off enveloping */
    pAnalParams->specEnvParams.iOrder = 25; /* ... but set default params anyway */
    pAnalParams->specEnvParams.fLambda = 0.00001;
    pAnalParams->specEnvParams.iMaxFreq = 0;
    pAnalParams->specEnvParams.nCoeff = 0;
    pAnalParams->specEnvParams.iAnchor = 0; /* not yet implemented */
    pAnalParams->pFrames = NULL;
    /* fft */
    for(i = 0; i < SMS_MAX_SPEC; i++)
    {
        pAnalParams->magSpectrum[i] = 0.0;
        pAnalParams->phaseSpectrum[i] = 0.0;
        pAnalParams->spectrumWindow[i] = 0.0;
        pAnalParams->fftBuffer[i] = 0.0;
        pAnalParams->fftBuffer[i+SMS_MAX_SPEC] = 0.0;
    }
    /* analysis frames */
    pAnalParams->pFrames = NULL;
    pAnalParams->ppFrames = NULL;
    /* residual */
    sms_initResidualParams(&pAnalParams->residualParams);
    /* peak continuation */
    pAnalParams->guideStates = NULL;
    pAnalParams->guides = NULL;
    /* audio input frame */
    for(i = 0; i < SMS_MAX_FRAME_SIZE; i++)
        pAnalParams->inputBuffer[i] = 0.0;
    /* stochastic analysis */
    pAnalParams->stocMagSpectrum = NULL;
    pAnalParams->approxEnvelope = NULL;
    pAnalParams->ppFrames = NULL;
}

/*! \brief initialize analysis data structure's arrays
 * 
 *  based on the SMS_AnalParams current settings, this function will
 *  initialize the sound, synth, and fft arrays. It is necessary before analysis.
 *  there can be multple SMS_AnalParams at the same time
 *
 * \param pAnalParams    pointer to analysis paramaters
 * \param pSoundHeader    pointer to sound header
 * \return 0 on success, -1 on error
 */
int sms_initAnalysis(SMS_AnalParams *pAnalParams)
{
    int i;
    SMS_SndBuffer *pSynthBuf = &pAnalParams->synthBuffer;
    SMS_SndBuffer *pSoundBuf = &pAnalParams->soundBuffer;

    /* define the hopsize for each record */
    pAnalParams->sizeHop = (int)(pAnalParams->iSamplingRate /
                                 (sfloat) pAnalParams->iFrameRate);

    /* set the default size window to an odd length */
    pAnalParams->iDefaultSizeWindow = 
        (int)((pAnalParams->iSamplingRate / pAnalParams->fDefaultFundamental) *
               pAnalParams->fSizeWindow / 2) * 2 + 1;

    int sizeBuffer = (pAnalParams->iMaxDelayFrames * pAnalParams->sizeHop) + SMS_MAX_WINDOW;

    /* if storing residual phases, restrict number of stochastic coefficients to the size of the spectrum (sizeHop = 1/2 sizeFft)*/
    if(pAnalParams->iStochasticType == SMS_STOC_IFFT)
        pAnalParams->nStochasticCoeff = sms_power2(pAnalParams->sizeHop);

    /* do the same if spectral envelope is to be stored in frequency bins */
    if(pAnalParams->specEnvParams.iType == SMS_ENV_FBINS)
        pAnalParams->specEnvParams.nCoeff = sms_power2(pAnalParams->specEnvParams.iOrder * 2);
    else if(pAnalParams->specEnvParams.iType == SMS_ENV_CEP)
        pAnalParams->specEnvParams.nCoeff = pAnalParams->specEnvParams.iOrder+1;
    /* if specEnvParams.iMaxFreq is still 0, set it to the same as fHighestFreq (normally what you want)*/
    if(pAnalParams->specEnvParams.iMaxFreq == 0)
        pAnalParams->specEnvParams.iMaxFreq = pAnalParams->fHighestFreq;

    /*\todo this probably doesn't need env coefficients - they aren't getting used */
    if(sms_allocFrame(&pAnalParams->prevFrame, pAnalParams->nGuides,
                      pAnalParams->nStochasticCoeff, 1, pAnalParams->iStochasticType, 0)
       == -1)
    {
        sms_error("Could not allocate memory for prevFrame");
        return -1;
    }

    pAnalParams->sizeNextRead = (pAnalParams->iDefaultSizeWindow + 1) * 0.5;

    /* sound buffer */
    if((pSoundBuf->pFBuffer = (sfloat *) calloc(sizeBuffer, sizeof(sfloat))) == NULL)
    {
        sms_error("Could not allocate memory for sound buffer");
        return -1;
    }
    pSoundBuf->iMarker = -sizeBuffer;
    pSoundBuf->iFirstGood = sizeBuffer;
    pSoundBuf->sizeBuffer = sizeBuffer;

    /* check default fundamental */
    if (pAnalParams->fDefaultFundamental < pAnalParams->fLowestFundamental)
    {
        pAnalParams->fDefaultFundamental = pAnalParams->fLowestFundamental;
    }
    if (pAnalParams->fDefaultFundamental > pAnalParams->fHighestFundamental)
    {
        pAnalParams->fDefaultFundamental = pAnalParams->fHighestFundamental;
    }

    /* deterministic synthesis buffer */
    pSynthBuf->sizeBuffer = pAnalParams->sizeHop << 1;
    pSynthBuf->pFBuffer = calloc(pSynthBuf->sizeBuffer, sizeof(sfloat));
    if(pSynthBuf->pFBuffer == NULL)
    {
        sms_error("could not allocate memory");
        return -1;
    }
    pSynthBuf->iMarker = pSynthBuf->sizeBuffer;
    /* buffer of analysis frames */
    pAnalParams->pFrames = (SMS_AnalFrame *)malloc(pAnalParams->iMaxDelayFrames * sizeof(SMS_AnalFrame));
    if(pAnalParams->pFrames == NULL)
    {
        sms_error("could not allocate memory for delay frames");
        return -1;
    }
    pAnalParams->ppFrames = (SMS_AnalFrame **)malloc(pAnalParams->iMaxDelayFrames * sizeof(SMS_AnalFrame *));
    if(pAnalParams->ppFrames == NULL)
    {
        sms_error("could not allocate memory for pointers to delay frames");
        return -1;
    }

    /* initialize the frame pointers and allocate memory */
    for(i = 0; i < pAnalParams->iMaxDelayFrames; i++)
    {
        pAnalParams->pFrames[i].iStatus = SMS_FRAME_EMPTY;
        pAnalParams->pFrames[i].iFrameSample = 0;
        pAnalParams->pFrames[i].iFrameSize = 0;
        pAnalParams->pFrames[i].iFrameNum = 0;
        pAnalParams->pFrames[i].pSpectralPeaks = 
            (SMS_Peak *)malloc(pAnalParams->maxPeaks * sizeof(SMS_Peak));
        if((pAnalParams->pFrames[i]).pSpectralPeaks == NULL)
        {
            sms_error("could not allocate memory for spectral peaks");
            return -1;
        }
        (pAnalParams->pFrames[i].deterministic).nTracks = pAnalParams->nGuides;

        (pAnalParams->pFrames[i].deterministic).pFSinFreq = 
            (sfloat *)calloc(pAnalParams->nGuides, sizeof(sfloat));
        if((pAnalParams->pFrames[i].deterministic).pFSinFreq == NULL)
        {
            sms_error("could not allocate memory");
            return -1;
        }

        (pAnalParams->pFrames[i].deterministic).pFSinAmp =
            (sfloat *)calloc(pAnalParams->nGuides, sizeof(sfloat));
        if((pAnalParams->pFrames[i].deterministic).pFSinAmp == NULL)
        {
            sms_error("could not allocate memory");
            return -1;
        }

        (pAnalParams->pFrames[i].deterministic).pFSinPha =
            (sfloat *)calloc(pAnalParams->nGuides, sizeof(sfloat));
        if((pAnalParams->pFrames[i].deterministic).pFSinPha == NULL)
        {
            sms_error("could not allocate memory");
            return -1;
        }
        pAnalParams->ppFrames[i] = &pAnalParams->pFrames[i];

        /* set initial values */
        if(sms_clearAnalysisFrame(i, pAnalParams) < 0)
        {
            sms_error("could not set initial values for analysis frames");
            return -1;
        }
    }

    /* memory for residual */
    pAnalParams->residualParams.residualSize = pAnalParams->sizeHop * 2;
    sms_initResidual(&pAnalParams->residualParams);

    /* memory for guide states */
    pAnalParams->guideStates = (int *)calloc(pAnalParams->nGuides, sizeof(int));
    if(pAnalParams->guideStates == NULL)
    {
        sms_error("Could not allocate memory for guide states");
        return -1;
    }

    /* memory for guides */
    pAnalParams->guides = (SMS_Guide *)malloc(pAnalParams->nGuides * sizeof(SMS_Guide));
    if(pAnalParams->guides == NULL)
    {
        sms_error("Could not allocate memory for guides");
        return -1;
    }

    /* initial guide values */
    for (i = 0; i < pAnalParams->nGuides; i++)
    {
        if(pAnalParams->iFormat == SMS_FORMAT_H || pAnalParams->iFormat == SMS_FORMAT_HP)
        {
            pAnalParams->guides[i].fFreq = pAnalParams->fDefaultFundamental * (i + 1);
        }
        else
        {
            pAnalParams->guides[i].fFreq = 0.0;
        }
        pAnalParams->guides[i].fMag = 0.0;
        pAnalParams->guides[i].iPeakChosen = -1;
        pAnalParams->guides[i].iStatus = 0;
    }

    /* stochastic analysis */
    pAnalParams->sizeStocMagSpectrum = sms_power2(pAnalParams->residualParams.residualSize) >> 1;
    pAnalParams->stocMagSpectrum = (sfloat *)calloc(pAnalParams->sizeStocMagSpectrum, sizeof(sfloat));
    if(pAnalParams->stocMagSpectrum == NULL)
    {
        sms_error("Could not allocate memory for stochastic magnitude spectrum");
        return -1;
    }
    pAnalParams->approxEnvelope = (sfloat *)calloc(pAnalParams->nStochasticCoeff, sizeof(sfloat));
    if(pAnalParams->approxEnvelope == NULL)
    {
        sms_error("Could not allocate memory for spectral approximation envelope");
        return -1;
    }

    return 0;
}

/*! \brief give default values to an SMS_SynthParams struct 
 * 
 * This will initialize an SMS_SynthParams with values that work
 * for common analyses.  It is useful to start with and then
 * adjust the parameters manually to fit a particular sound
 *
 * \param synthParams    pointer to synthesis parameters data structure
 */
void sms_initSynthParams(SMS_SynthParams *synthParams)
{
    synthParams->iSamplingRate = 44100;
    synthParams->iOriginalSRate = 44100;
    synthParams->iSynthesisType = SMS_STYPE_ALL;
    synthParams->iDetSynthType = SMS_DET_IFFT;
    synthParams->sizeHop = SMS_MIN_SIZE_FRAME;
    synthParams->origSizeHop = SMS_MIN_SIZE_FRAME;
    synthParams->nTracks = 60;
    synthParams->iStochasticType = SMS_STOC_APPROX;
    synthParams->nStochasticCoeff = 128;
    synthParams->pFDetWindow = NULL;
    synthParams->pFStocWindow = NULL;
    synthParams->pSynthBuff = NULL;
    synthParams->pMagBuff = NULL;
    synthParams->pPhaseBuff = NULL;
    synthParams->pSpectra = NULL;
    synthParams->approxEnvelope = NULL;
    synthParams->deEmphasis = 1; /*!< perform de-emphasis by default */
    synthParams->deEmphasisLastValue = 0;
}

/*! \brief initialize synthesis data structure's arrays
 * 
 *  Initialize the synthesis and fft arrays. It is necessary before synthesis.
 *  there can be multple SMS_SynthParams at the same time
 *  This function also sets some initial values that will create a sane synthesis
 *  environment.
 *
 * This function requires an SMS_Header because it may be called to synthesize
 * a stored .sms file, which contains a header with necessary information.
 *
 * \param pSmsHeader      pointer to SMS_Header
 * \param pSynthParams    pointer to synthesis paramaters
 * \return 0 on success, -1 on error
 */
int sms_initSynth(SMS_SynthParams *pSynthParams)
{
    int sizeHop, sizeFft;

    /* make sure sizeHop is something to the power of 2 */
    sizeHop = sms_power2(pSynthParams->sizeHop);
    if(sizeHop != pSynthParams->sizeHop)
    {
        printf("Warning: Synthesis hop size (%d) was not a power of two.\n",
                pSynthParams->sizeHop);
        printf("         Changed to %d.\n", sizeHop);
        pSynthParams->sizeHop = sizeHop;
    }
    sizeFft = sizeHop * 2;

    /* TODO: check memory allocation */
    pSynthParams->pFStocWindow = (sfloat *)calloc(sizeFft, sizeof(sfloat));
    sms_getWindow(sizeFft, pSynthParams->pFStocWindow, SMS_WIN_HANNING);
    pSynthParams->pFDetWindow = (sfloat *)calloc(sizeFft, sizeof(sfloat));
    sms_getWindow(sizeFft, pSynthParams->pFDetWindow, SMS_WIN_IFFT);

    /* allocate memory for analysis data - size of original hopsize 
     * previous frame to interpolate from */
    /* \todo why is stoch coeff + 1? */
    sms_allocFrame(&pSynthParams->prevFrame, pSynthParams->nTracks,
                   pSynthParams->nStochasticCoeff + 1, 1,
                   pSynthParams->iStochasticType, 0);

    pSynthParams->pSynthBuff = (sfloat *)calloc(sizeFft, sizeof(sfloat));
    pSynthParams->pMagBuff = (sfloat *)calloc(sizeHop, sizeof(sfloat));
    pSynthParams->pPhaseBuff = (sfloat *)calloc(sizeHop, sizeof(sfloat));
    pSynthParams->pSpectra = (sfloat *)calloc(sizeFft, sizeof(sfloat));

    /* approximation envelope */
    pSynthParams->approxEnvelope = (sfloat *)calloc(pSynthParams->nStochasticCoeff, sizeof(sfloat));
    if(pSynthParams->approxEnvelope == NULL)
    {
        sms_error("Could not allocate memory for spectral approximation envelope");
        return -1;
    }

    return SMS_OK;
}

/*! \brief give default values to an SMS_ResidualParams struct 
 * 
 * \param residualParams pointer to residual data structure
 */
void sms_initResidualParams(SMS_ResidualParams *residualParams)
{
    residualParams->samplingRate = 44100;
    residualParams->residualSize = 0;
    residualParams->residual = NULL;
    residualParams->residualWindow = NULL;
    residualParams->residualMag = 0.0;
    residualParams->originalMag = 0.0;
    residualParams->nCoeffs = 128;
    residualParams->stocCoeffs = NULL;
    residualParams->sizeStocMagSpectrum = 0;
    residualParams->stocMagSpectrum = NULL;
    residualParams->approxEnvelope = NULL;
    int i;
    for(i = 0; i < SMS_MAX_SPEC; i++)
    {
        residualParams->fftBuffer[i] = 0.0;
        residualParams->fftBuffer[i+SMS_MAX_SPEC] = 0.0;
    }
}

/*! \brief initialize residual data structure
 * 
 * \param residualParams pointer to synthesis paramaters
 * \return 0 on success, -1 on error
 */
int sms_initResidual(SMS_ResidualParams *residualParams)
{
    if(residualParams->residualSize <= 0)
    {
        sms_error("Residual size must be a positive integer");
        return -1;
    }

    /* residual signal */
    residualParams->residual = (sfloat *)calloc(residualParams->residualSize, sizeof(sfloat));
    if(residualParams->residual == NULL)
    {
        sms_error("Could not allocate memory for residual");
        return -1;
    }

    /* residual window */
    residualParams->residualWindow = (sfloat *)calloc(residualParams->residualSize, sizeof(sfloat));
    if(residualParams->residualWindow == NULL)
    {
        sms_error("Could not allocate memory for residualWindow");
        return -1;
    }
    sms_getWindow(residualParams->residualSize, residualParams->residualWindow, SMS_WIN_HAMMING);
    sms_scaleWindow(residualParams->residualSize, residualParams->residualWindow);

    /* stochastic analysis */
    residualParams->stocCoeffs = (sfloat *)calloc(residualParams->nCoeffs, sizeof(sfloat));
    if(residualParams->stocCoeffs == NULL)
    {
        sms_error("Could not allocate memory for stochastic coefficients");
        return -1;
    }

    residualParams->sizeStocMagSpectrum = sms_power2(residualParams->residualSize) >> 1;
    residualParams->stocMagSpectrum = (sfloat *)calloc(residualParams->sizeStocMagSpectrum, sizeof(sfloat));
    if(residualParams->stocMagSpectrum == NULL)
    {
        sms_error("Could not allocate memory for stochastic magnitude spectrum");
        return -1;
    }

    residualParams->approxEnvelope = (sfloat *)calloc(residualParams->nCoeffs, sizeof(sfloat));
    if(residualParams->approxEnvelope == NULL)
    {
        sms_error("Could not allocate memory for spectral approximation envelope");
        return -1;
    }

    return 0;
}

/*! \brief free residual data
 * 
 * frees all the memory allocated to an SMS_ResidualParams by
 * sms_initResidual
 *
 * \param residualParams pointer to residual data structure
 */
void sms_freeResidual(SMS_ResidualParams *residualParams)
{
    if(residualParams->residual)
        free(residualParams->residual);
    if(residualParams->residualWindow)
        free(residualParams->residualWindow);
    if(residualParams->stocCoeffs)
        free(residualParams->stocCoeffs);
    if(residualParams->stocMagSpectrum)
        free(residualParams->stocMagSpectrum);
    if(residualParams->approxEnvelope)
        free(residualParams->approxEnvelope);

    residualParams->residual = NULL;
    residualParams->residualWindow = NULL;
    residualParams->stocCoeffs = NULL;
    residualParams->stocMagSpectrum = NULL;
    residualParams->approxEnvelope = NULL;
}

/*! \brief free analysis data
 * 
 * frees all the memory allocated to an SMS_AnalParams by
 * sms_initAnalysis
 *
 * \param pAnalParams    pointer to analysis data structure
 */
void sms_freeAnalysis(SMS_AnalParams *pAnalParams)
{
    if(pAnalParams->pFrames)
    {
        int i;
        for(i = 0; i < pAnalParams->iMaxDelayFrames; i++)
        {
            if((pAnalParams->pFrames[i]).pSpectralPeaks)
                free((pAnalParams->pFrames[i]).pSpectralPeaks);
            if((pAnalParams->pFrames[i].deterministic).pFSinFreq)
               free((pAnalParams->pFrames[i].deterministic).pFSinFreq);
            if((pAnalParams->pFrames[i].deterministic).pFSinAmp)
               free((pAnalParams->pFrames[i].deterministic).pFSinAmp);
            if((pAnalParams->pFrames[i].deterministic).pFSinPha)
               free((pAnalParams->pFrames[i].deterministic).pFSinPha);
        }
        free(pAnalParams->pFrames);
    }

    sms_freeFrame(&pAnalParams->prevFrame);
    sms_freeResidual(&pAnalParams->residualParams);

    if(pAnalParams->soundBuffer.pFBuffer)
        free(pAnalParams->soundBuffer.pFBuffer);
    if((pAnalParams->synthBuffer).pFBuffer)
        free((pAnalParams->synthBuffer).pFBuffer);
    if(pAnalParams->ppFrames)
        free(pAnalParams->ppFrames);
    if(pAnalParams->guideStates)
        free(pAnalParams->guideStates);
    if(pAnalParams->guides)
        free(pAnalParams->guides);
    if(pAnalParams->stocMagSpectrum)
        free(pAnalParams->stocMagSpectrum);
    if(pAnalParams->approxEnvelope)
        free(pAnalParams->approxEnvelope);

    pAnalParams->pFrames = NULL;
    pAnalParams->ppFrames = NULL;
    pAnalParams->soundBuffer.pFBuffer = NULL;
    pAnalParams->synthBuffer.pFBuffer = NULL;
    pAnalParams->guideStates = NULL;
    pAnalParams->guides = NULL;
    pAnalParams->stocMagSpectrum = NULL;
    pAnalParams->approxEnvelope = NULL;
}

/*! \brief free analysis data
 * 
 * frees all the memory allocated to an SMS_SynthParams by
 * sms_initSynthesis
 *
 * \todo is there a way to make sure the plan has been made
 * already? as it is, it crashes if this is called without one
 * \param pSynthParams    pointer to synthesis data structure
 */
void sms_freeSynth(SMS_SynthParams *pSynthParams)
{
    if(pSynthParams->pFStocWindow)
        free(pSynthParams->pFStocWindow);        
    if(pSynthParams->pFDetWindow)
        free(pSynthParams->pFDetWindow);
    if(pSynthParams->pSynthBuff)
        free(pSynthParams->pSynthBuff);
    if(pSynthParams->pSpectra)
        free(pSynthParams->pSpectra);
    if(pSynthParams->pMagBuff)
        free(pSynthParams->pMagBuff);
    if(pSynthParams->pPhaseBuff)
        free(pSynthParams->pPhaseBuff);
    if(pSynthParams->approxEnvelope)
        free(pSynthParams->approxEnvelope);

    sms_freeFrame(&pSynthParams->prevFrame);
}

/*! \brief Allocate memory for an array of spectral peaks
 *
 * Creates memory and sets default values.
 *
 * \param peaks the spectral peaks
 * \param n number of peaks
 * \return 0 on success, -1 on error
 */
int sms_initSpectralPeaks(SMS_SpectralPeaks* peaks, int n)
{
    peaks->nPeaks = n;
    peaks->nPeaksFound = 0;

    peaks->pSpectralPeaks = (SMS_Peak *)malloc(n * sizeof(SMS_Peak));
    if(peaks->pSpectralPeaks == NULL)
    {
        sms_error("could not allocate memory for spectral peaks");
        return -1;
    }
    return 0;
}

/*! \brief Deallocate memory for an array of spectral peaks
 *
 * \param peaks the spectral peaks
 */
void sms_freeSpectralPeaks(SMS_SpectralPeaks* peaks)
{
    if(!peaks)
        return;

    if(peaks->pSpectralPeaks)
        free(peaks->pSpectralPeaks);
    peaks->nPeaks = 0;
    peaks->nPeaksFound = 0;
}

/*! \brief set window size for next frame 
 *
 * adjusts the next window size to fit the currently detected fundamental 
 * frequency, or resets to a default window size if unstable.
 *
 * \param iCurrentFrame         number of current frame
 * \param pAnalParams          analysis parameters
 * \return the size of the next window in samples
 */
int sms_sizeNextWindow(int iCurrentFrame, SMS_AnalParams *pAnalParams)
{
    sfloat fFund = pAnalParams->ppFrames[iCurrentFrame]->fFundamental;
    sfloat fPrevFund = pAnalParams->ppFrames[iCurrentFrame-1]->fFundamental;
    int sizeWindow;

    /* if the previous fundamental was stable use it to set the window size */
    if(fPrevFund > 0 && fabs(fPrevFund - fFund) / fFund <= .2)
        sizeWindow = (int)((pAnalParams->iSamplingRate / fFund) *
                           pAnalParams->fSizeWindow * .5) * 2 + 1;
    /* otherwise use the default size window */
    else
        sizeWindow = pAnalParams->iDefaultSizeWindow;

    if(sizeWindow > SMS_MAX_WINDOW)
    {
        fprintf(stderr, "sms_sizeNextWindow error: sizeWindow (%d) too big, set to %d\n", sizeWindow, 
                SMS_MAX_WINDOW);
        sizeWindow = SMS_MAX_WINDOW;
    }

    return sizeWindow;
}

/*! \brief set default values for analysis frame variables
 * \param iCurrentFrame frame number of the current frame
 * \param pAnalParams analysis parameters
 * \return 0 on success, -1 on error
 */
int sms_clearAnalysisFrame(int iCurrentFrame, SMS_AnalParams *pAnalParams)
{
    int i;
    SMS_AnalFrame *currentFrame = pAnalParams->ppFrames[iCurrentFrame];

    /* clear deterministic data */
    for(i = 0; i < pAnalParams->nGuides; i++)
    {
        currentFrame->deterministic.pFSinFreq[i] = 0.0;
        currentFrame->deterministic.pFSinAmp[i] = 0.0;
        currentFrame->deterministic.pFSinPha[i] = 0.0;
    }

    /* clear peaks */
    for(i = 0; i < pAnalParams->maxPeaks; i++)
    {
       currentFrame->pSpectralPeaks[i].fFreq = 0.0;
       currentFrame->pSpectralPeaks[i].fMag = 0.0;
       currentFrame->pSpectralPeaks[i].fPhase = 0.0;
    }

    currentFrame->nPeaks = 0;
    currentFrame->fFundamental = 0;
    currentFrame->iFrameNum = 0;
    currentFrame->iFrameSize = 0;
    currentFrame->iFrameSample = 0;
    currentFrame->iStatus = SMS_FRAME_EMPTY;

    return 0;
}

/*! \brief initialize the current frame
 *
 * initializes arrays to zero and sets the correct sample position.
 * Special care is taken at the end the sample source (if there is
 * not enough samples for an entire frame.
 *
 * \param iCurrentFrame            frame number of current frame in buffer
 * \param pAnalParams             analysis parameters
 * \param sizeWindow               size of analysis window 
 * \return -1 on error \todo make this return void
 */
int sms_initFrame(int iCurrentFrame, SMS_AnalParams *pAnalParams, int sizeWindow)
{
    /* clear deterministic data */
    memset((sfloat *)pAnalParams->ppFrames[iCurrentFrame]->deterministic.pFSinFreq, 0, 
           sizeof(sfloat) * pAnalParams->nGuides);
    memset((sfloat *)pAnalParams->ppFrames[iCurrentFrame]->deterministic.pFSinAmp, 0, 
           sizeof(sfloat) * pAnalParams->nGuides);
    memset((sfloat *)pAnalParams->ppFrames[iCurrentFrame]->deterministic.pFSinPha, 0, 
           sizeof(sfloat) * pAnalParams->nGuides);

    /* clear peaks */
    int i;
    for(i = 0; i < pAnalParams->maxPeaks; i++)
    {
        pAnalParams->ppFrames[iCurrentFrame]->pSpectralPeaks[i].fFreq = 0.0;
        pAnalParams->ppFrames[iCurrentFrame]->pSpectralPeaks[i].fMag = 0.0;
        pAnalParams->ppFrames[iCurrentFrame]->pSpectralPeaks[i].fPhase = 0.0;
    }

    pAnalParams->ppFrames[iCurrentFrame]->nPeaks = 0;
    pAnalParams->ppFrames[iCurrentFrame]->fFundamental = 0;

    pAnalParams->ppFrames[iCurrentFrame]->iFrameNum =  
        pAnalParams->ppFrames[iCurrentFrame - 1]->iFrameNum + 1;
    pAnalParams->ppFrames[iCurrentFrame]->iFrameSize = sizeWindow;

    /* if first frame set center of data around 0 */
    if(pAnalParams->ppFrames[iCurrentFrame]->iFrameNum == 1)
        pAnalParams->ppFrames[iCurrentFrame]->iFrameSample = 0;
    /* if not, increment center of data by sizeHop */
    else
        pAnalParams->ppFrames[iCurrentFrame]->iFrameSample = 
            pAnalParams->ppFrames[iCurrentFrame-1]->iFrameSample + pAnalParams->sizeHop;

    /* check for end of sound */
    if((pAnalParams->ppFrames[iCurrentFrame]->iFrameSample + (sizeWindow+1)/2) >= pAnalParams->iSizeSound
         && pAnalParams->iSizeSound > 0)
    {
        pAnalParams->ppFrames[iCurrentFrame]->iFrameNum =  -1;
        pAnalParams->ppFrames[iCurrentFrame]->iFrameSize =  0;
        pAnalParams->ppFrames[iCurrentFrame]->iStatus =  SMS_FRAME_END;
    }
    else
    {
        /* good status, ready to start computing */
        pAnalParams->ppFrames[iCurrentFrame]->iStatus = SMS_FRAME_READY;
    }
    return SMS_OK;
}

/*! \brief get deviation from average fundamental
 *\
 * \param pAnalParams             pointer to analysis params
 * \param iCurrentFrame        number of current frame 
 * \return deviation value or -1 if really off
 */
sfloat sms_fundDeviation(SMS_AnalParams *pAnalParams, int iCurrentFrame)
{
    sfloat fFund, fSum = 0, fAverage, fDeviation = 0;
    int i;

    if(pAnalParams->minGoodFrames < 1)
        return -1;

    /* get the sum of the past few fundamentals */
    for(i = 0; (i < pAnalParams->minGoodFrames) && (iCurrentFrame-i >= 0); i++)
    {
        fFund = pAnalParams->ppFrames[iCurrentFrame-i]->fFundamental;
        if(fFund <= 0)
            return -1;
        else
            fSum += fFund;
    }

    /* find the average */
    fAverage = fSum / pAnalParams->minGoodFrames;

    /* get the deviation from the average */
    for(i = 0; (i < pAnalParams->minGoodFrames) && (iCurrentFrame-i >= 0); i++)
        fDeviation += fabs(pAnalParams->ppFrames[iCurrentFrame-i]->fFundamental - fAverage);

    /* return the deviation from the average */
    return fDeviation / (pAnalParams->minGoodFrames * fAverage);
}


/*! \brief function to create the debug file 
 *
 * \param pAnalParams             pointer to analysis params
 * \return error value \see SMS_ERRORS 
 */
int sms_createDebugFile(SMS_AnalParams *pAnalParams)
{
    if((pDebug = fopen(pChDebugFile, "w+")) == NULL) 
    {
        fprintf(stderr, "Cannot open debugfile: %s\n", pChDebugFile);
        return SMS_WRERR;
    }
    return SMS_OK;
}

/*! \brief  function to write to the debug file
 *
 * writes three arrays of equal size to a debug text
 * file ("./debug.txt"). There are three arrays for the 
 * frequency, magnitude, phase sets. 
 * 
 * \param pFBuffer1 pointer to array 1
 * \param pFBuffer2 pointer to array 2
 * \param pFBuffer3 pointer to array 3
 * \param sizeBuffer the size of the buffers
 */
void sms_writeDebugData(sfloat *pFBuffer1, sfloat *pFBuffer2, 
                        sfloat *pFBuffer3, int sizeBuffer)
{
    int i;
    static int counter = 0;

    for(i = 0; i < sizeBuffer; i++)
        fprintf(pDebug, "%d %d %d %d\n", counter++, (int)pFBuffer1[i],
               (int)pFBuffer2[i], (int)pFBuffer3[i]);
}

/*! \brief  function to write the residual sound file to disk
 *
 * writes the "debug.txt" file to disk and closes the file.
 */
void sms_writeDebugFile ()
{
    fclose(pDebug);
}

/*! \brief convert from magnitude to decibel
 *
 * \param x      magnitude (0:1)
 * \return         decibel (0: -100)
 */
sfloat sms_magToDB(sfloat x)
{
    if(x < mag_thresh)
        return 0.0;
    else
        //return(20. * log10(x * inv_mag_thresh));
        return(TWENTY_OVER_LOG10 * log(x * inv_mag_thresh));
        /*return(TWENTY_OVER_LOG10 * log(x));*/
}

/*! \brief convert from decibel to magnitude
 *
 * \param x     decibel (0-100)
 * \return        magnitude (0-1)
 */
sfloat sms_dBToMag(sfloat x)
{
    if(x < 0.00001)
        return 0.0;
    else
        return(mag_thresh * pow(10., x*0.05));
        /*return pow(10.0, x*0.05);*/
}

/*! \brief convert an array from magnitude to decibel 
 *
 * Depends on a  linear threshold that indicates the bottom end
 * of the dB scale (magnutdes at this value will convert to zero).
 * \see sms_setMagThresh
 *
 * \param sizeArray     size of array
 * \param pArray pointer to array
 */
void sms_arrayMagToDB(int sizeArray, sfloat *pArray)
{
    int i;
    for(i = 0; i < sizeArray; i++)
        pArray[i] = sms_magToDB(pArray[i]);
}

/*! \brief convert and array from decibel (0-100) to magnitude (0-1)
 *
 * depends on the magnitude threshold
 * \see sms_setMagThresh
 *
 * \param sizeArray     size of array
 * \param pArray pointer to array
 */
void sms_arrayDBToMag(int sizeArray, sfloat *pArray)
{
    int i;
    for(i = 0; i < sizeArray; i++)
        pArray[i] = sms_dBToMag(pArray[i]);
}
/*! \brief set the linear magnitude threshold
 *
 * magnitudes below this will go to zero when converted to db.
 * it is limited to 0.00001 (-100db)
 *
 * \param x  threshold value
 */
void sms_setMagThresh(sfloat x)
{
    /* limit threshold to -100db */
    if(x < 0.00001) 
        mag_thresh = 0.00001;
    else
        mag_thresh = x;
    inv_mag_thresh = 1. / mag_thresh;
}

/*! \brief get a string containing information about the error code 
 *
 * \param pErrorMessage pointer to error message string
 */
void sms_error(char *pErrorMessage) 
{
    strncpy(error_message, pErrorMessage, 256);
    error_status = -1;
}

/*! \brief check if an error has been reported
 *
 * \return  -1 if there is an error, 0 if ok
 */
int sms_errorCheck() 
{
    return error_status;
}

/*! \brief get a string containing information about the last error 
 *
 * \return  pointer to a char string, or NULL if no error
 */
char* sms_errorString() 
{
    if (error_status)
    {
        error_status = 0;
        return error_message;
    }
    return NULL;
}

/*! \brief random number genorator
 *
 * \return random number between -1 and 1
 */
sfloat sms_random()
{
#ifdef MERSENNE_TWISTER
    return genrand_real1(); 
#else
    return (sfloat)(random() * 2 * INV_HALF_MAX);
#endif
}

/*! \brief Root Mean Squared of an array
 *
 * \return RMS energy
 */
sfloat sms_rms(int sizeArray, sfloat *pArray)
{
    int i;
    sfloat mean_squared = 0.;
    for(i = 0; i < sizeArray; i++)
        mean_squared += pArray[i] * pArray[i];

    return sqrtf(mean_squared / sizeArray);
}

/*! \brief make sure a number is a power of 2
 *
 * \return a power of two integer >= input value
 */
int sms_power2(int n)
{
    int p = -1;
    int N = n;
    while(n)
    {
        n >>= 1;
        p++;
    }

    if(1<<p == N) /* n was a power of 2 */
    {
        return N; 
    }
    else  /* make the new value larger than n */
    {
        p++;
        return 1<<p;
    }
}

/*! \brief compute a value for scaling frequency based on the well-tempered scale
 *
 * \param x linear frequency value
 * \return (1.059...)^x, where 1.059 is the 12th root of 2 precomputed
 */
sfloat sms_scalarTempered(sfloat x)
{
    return powf(1.0594630943592953, x);
}

/*! \brief scale an array of linear frequencies to the well-tempered scale
 *
 * \param sizeArray size of the array
 * \param pArray pointer to array of frequencies
 */
void sms_arrayScalarTempered(int sizeArray, sfloat *pArray)
{
    int i;
    for(i = 0; i < sizeArray; i++)
        pArray[i] = sms_scalarTempered(pArray[i]);
}

