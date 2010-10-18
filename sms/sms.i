%module pysms
%{
    #include "sms.h"
    #define SWIG_FILE_WITH_INIT
%}

%include "../common/numpy.i"

%init 
%{
    import_array(); 
    sms_init(); /* initialize the library (makes some tables, seeds the random number generator, etc */
%}

%exception
{
    $action
    if (sms_errorCheck())
    {
        PyErr_SetString(PyExc_IndexError,sms_errorString());
        return NULL;
    }
}

/* apply all numpy typemaps to various names in sms.h */
%apply (int DIM1, float* INPLACE_ARRAY1) {(int sizeWindow, float* pWindow)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(int sizeWaveform, float* pWaveform)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(long sizeSound, float* pSound)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(int sizeFft, float* pArray)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(int sizeFft, float* pFftBuffer)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(int sizeFreq, float* pFreq)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(int sizeAmp, float* pAmp)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(int sizeMag, float* pMag)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(int sizePhase, float* pPhase)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(int sizeRes, float* pRes)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(int sizeCepstrum, float* pCepstrum)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(int sizeEnv, float* pEnv)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(int sizeTrack, float* pTrack)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(int sizeArray, float* pArray)};
%apply (int DIM1, float* IN_ARRAY1) {(int sizeInArray, float* pInArray)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(int sizeOutArray, float* pOutArray)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(int sizeHop, float* pSynthesis)};
%apply (int DIM1, float* IN_ARRAY1)
{
    (int numamps, float* amps),
    (int numfreqs, float* freqs),
    (int numphases, float* phases)
}
%apply (int DIM1, float* IN_ARRAY1)
{
    (int sizeSynthesis, float* pSynthesis),
    (int sizeOriginal, float* pOriginal),
    (int sizeResidual, float* pResidual)
}

%include "sms.h" 


/* overload the functions that will be wrapped to fit numpy typmaps (defined below)
 * by renaming the wrapped names back to originals 
 */
%rename (sms_detectPeaks) simplsms_detectPeaks; 
%rename (sms_spectrum) simplsms_spectrum; 
%rename (sms_spectrumMag) simplsms_spectrumMag; 
%rename (sms_windowCentered) simplsms_windowCentered; 
%rename (sms_invSpectrum) simplsms_invSpectrum; 
%rename (sms_dCepstrum) simplsms_dCepstrum;
%rename (sms_synthesize) simplsms_synthesize_wrapper;  

%inline 
%{
	typedef struct 
	{
		SMS_Header *header;
		SMS_Data *smsData;
		int allocated;
	} SMS_File;
	
	void simplsms_dCepstrum( int sizeCepstrum, float *pCepstrum, int sizeFreq, float *pFreq, int sizeMag, float *pMag, 
						float fLambda, int iSamplingRate)
	{
		sms_dCepstrum(sizeCepstrum,pCepstrum, sizeFreq, pFreq, pMag, 
					  fLambda, iSamplingRate);
	}
	int simplsms_detectPeaks(int sizeMag, float *pMag, int sizePhase, float *pPhase, 
						  SMS_SpectralPeaks *pPeakStruct, SMS_PeakParams *pPeakParams)
	{
		if(sizeMag != sizePhase)
		{ 
			sms_error("sizeMag != sizePhase");
			return 0;
		}
		if(pPeakStruct->nPeaks < pPeakParams->iMaxPeaks)
		{ 
			sms_error("nPeaks in SMS_SpectralPeaks is not large enough (less than SMS_PeakParams.iMaxPeaks)");
			return 0;
		}
		pPeakStruct->nPeaksFound = sms_detectPeaks(sizeMag, pMag, pPhase, pPeakStruct->pSpectralPeaks, pPeakParams);
		return pPeakStruct->nPeaksFound;	
	}
	int simplsms_spectrum( int sizeWaveform, float *pWaveform, int sizeWindow, float *pWindow,
						int sizeMag, float *pMag, int sizePhase, float *pPhase)
	{
		return(sms_spectrum(sizeWindow, pWaveform, pWindow, sizeMag, pMag, pPhase));
	}
	int simplsms_spectrumMag( int sizeWaveform, float *pWaveform, int sizeWindow, float *pWindow,
						   int sizeMag, float *pMag)
	{
		return(sms_spectrumMag(sizeWindow, pWaveform, pWindow, sizeMag, pMag));
	}
	int simplsms_invSpectrum( int sizeWaveform, float *pWaveform, int sizeWindow, float *pWindow,
						   int sizeMag, float *pMag, int sizePhase, float *pPhase)
	{
		return(sms_invSpectrum(sizeWaveform, pWaveform, pWindow, sizeMag, pMag, pPhase));
	}
	void simplsms_windowCentered(int sizeWaveform, float *pWaveform, int sizeWindow,
							  float *pWindow, int sizeFft, float *pFftBuffer)
	{
		if (sizeWaveform != sizeWindow)
		{ 
			sms_error("sizeWaveform != sizeWindow");
			return;
		}
		sms_windowCentered(sizeWindow, pWaveform, pWindow, sizeFft, pFftBuffer);
	}
	void simplsms_synthesize_wrapper(SMS_Data *pSmsData, int sizeHop, float *pSynthesis, SMS_SynthParams *pSynthParams) 
	{
		if(sizeHop != pSynthParams->sizeHop)
		{
			sms_error("sizeHop != pSynthParams->sizeHop");
			return;
		}
		sms_synthesize(pSmsData, pSynthesis, pSynthParams);
	}

%}

%extend SMS_File 
{
        /* load an entire file to an internal numpy array */
        void load( char *pFilename )
        {
                int i;
                FILE *pSmsFile;
                $self->allocated = 0;
                sms_getHeader (pFilename, &$self->header, &pSmsFile);
                if(sms_errorCheck()) return;
                
                $self->smsData = calloc($self->header->nFrames, sizeof(SMS_Data));
                for( i = 0; i < $self->header->nFrames; i++ )
                {
                        sms_allocFrameH ($self->header,  &$self->smsData[i]);
                        if(sms_errorCheck()) return;
                        sms_getFrame (pSmsFile, $self->header, i, &$self->smsData[i]);
                        if(sms_errorCheck()) return;
                }
                $self->allocated = 1;
                return;
        }
        void close(void) /* todo: this should be in the destructor, no? */
        {
                int i;
                if(!$self->allocated)
                {
                        sms_error("file not yet alloceted");
                        return;
                }
                $self->allocated = 0;
                for( i = 0; i < $self->header->nFrames; i++)
                        sms_freeFrame(&$self->smsData[i]);
                free($self->smsData);
                return;
        }
        /* return a pointer to a frame, which can be passed around to other libsms functions */
        void getFrame(int i, SMS_Data *frame)
        {
                if(i < 0 || i >= $self->header->nFrames)
                {
                        sms_error("index is out of file boundaries");
                        return;
                }
                frame = &$self->smsData[i];
        }
        void getTrack(int track, int sizeFreq, float *pFreq, int sizeAmp,
                   float *pAmp)
        {
                /* fatal error protection first */
                if(!$self->allocated)
                {
                        sms_error("file not yet alloceted");
                        return ;
                }
                if(track >= $self->header->nTracks)
                {
                        sms_error("desired track is greater than number of tracks in file");
                        return;
                }
                if(sizeFreq != sizeAmp)
                {
                        sms_error("freq and amp arrays are different in size");
                        return;
                }
                /* make sure arrays are big enough, or return less data */
                int nFrames = MIN (sizeFreq, $self->header->nFrames);
                int i;
                for( i=0; i < nFrames; i++)
                {
                        pFreq[i] = $self->smsData[i].pFSinFreq[track];
                        pAmp[i] = $self->smsData[i].pFSinAmp[track];
                }
        }
        // TODO turn into getTrackP - and check if phase exists
        void getTrack(int track, int sizeFreq, float *pFreq, int sizeAmp,
                   float *pAmp, int sizePhase, float *pPhase)
        {
                /* fatal error protection first */
                if(!$self->allocated)
                {
                        sms_error("file not yet alloceted");
                        return ;
                }
                if(track >= $self->header->nTracks)
                {
                        sms_error("desired track is greater than number of tracks in file");
                        return;
                }
                if(sizeFreq != sizeAmp)
                {
                        sms_error("freq and amp arrays are different in size");
                        return;
                }
                /* make sure arrays are big enough, or return less data */
                int nFrames = MIN (sizeFreq, $self->header->nFrames);
                int i;
                for( i=0; i < nFrames; i++)
                {
                        pFreq[i] = $self->smsData[i].pFSinFreq[track];
                        pAmp[i] = $self->smsData[i].pFSinFreq[track];
                }
                if($self->header->iFormat < SMS_FORMAT_HP) return;
                
                if(sizePhase != sizeFreq || sizePhase != sizeAmp)
                {
                        sms_error("phase array and freq/amp arrays are different in size");
                        return;
                }
                for( i=0; i < nFrames; i++)
                        pPhase[i] = $self->smsData[i].pFSinPha[track];
                
                return;
        }
        void getFrameDet(int i, int sizeFreq, float *pFreq, int sizeAmp, float *pAmp)
        {
                if(!$self->allocated)
                {
                        sms_error("file not yet alloceted");
                        return ;
                }
                if(i >= $self->header->nFrames)
                {
                        sms_error("index is greater than number of frames in file");
                        return;
                }
                int nTracks = $self->smsData[i].nTracks;
                if(sizeFreq > nTracks)
                {
                        sms_error("index is greater than number of frames in file");
                        return;
                }
                if(sizeFreq != sizeAmp)
                {
                        sms_error("freq and amp arrays are different in size");
                        return;
                }
                memcpy(pFreq, $self->smsData[i].pFSinFreq, sizeof(float) * nTracks);
                memcpy(pAmp, $self->smsData[i].pFSinAmp, sizeof(float) * nTracks);
                
                if($self->header->iFormat < SMS_FORMAT_HP) return;
                
                return;
        }
        void getFrameDetP(int i, int sizeFreq, float *pFreq, int sizeAmp,
                             float *pAmp, int sizePhase, float *pPhase)
        {
                if(!$self->allocated)
                {
                        sms_error("file not yet alloceted");
                        return ;
                }
                if($self->header->iFormat < SMS_FORMAT_HP) 
                {
                        sms_error("file does not contain a phase component in Deterministic (iFormat < SMS_FORMAT_HP)");
                        return;
                }
                if(i >= $self->header->nFrames)
                {
                        sms_error("index is greater than number of frames in file");
                        return;
                }
                int nTracks = $self->smsData[i].nTracks;
                if(sizeFreq > nTracks)
                {
                        sms_error("index is greater than number of frames in file");
                        return;
                }
                if(sizeFreq != sizeAmp)
                {
                        sms_error("freq and amp arrays are different in size");
                        return;
                }
                memcpy(pFreq, $self->smsData[i].pFSinFreq, sizeof(float) * nTracks);
                memcpy(pAmp, $self->smsData[i].pFSinAmp, sizeof(float) * nTracks);
                
                if(sizePhase != sizeFreq || sizePhase != sizeAmp)
                {
                        sms_error("phase array and freq/amp arrays are different in size");
                        return;
                }
                memcpy(pPhase, $self->smsData[i].pFSinPha, sizeof(float) * nTracks);
                
                return;
        }
        void getFrameRes(int i, int sizeRes, float *pRes)
        {
                
                if(!$self->allocated)
                {
                        sms_error("file not yet alloceted");
                        return ;
                }
                if($self->header->iStochasticType < 1) 
                {
                        sms_error("file does not contain a stochastic component");
                        return ;
                }
                int nCoeff = sizeRes;
                if($self->header->nStochasticCoeff > sizeRes) 
                        nCoeff = $self->header->nStochasticCoeff; // return what you can

                memcpy(pRes, $self->smsData[i].pFStocCoeff, sizeof(float) * nCoeff);
                return;
        }
        void getFrameEnv(int i, int sizeEnv, float *pEnv)
        {
                
                if(!$self->allocated)
                {
                        sms_error("file not yet alloceted");
                        return ;
                }
                if($self->header->iEnvType < 1) 
                {
                        sms_error("file does not contain a spectral envelope");
                        return ;
                }
                int nCoeff = sizeEnv;
                if($self->header->nStochasticCoeff > sizeEnv) 
                        nCoeff = $self->header->nEnvCoeff; // return what you can

                memcpy(pEnv, $self->smsData[i].pSpecEnv, sizeof(sfloat) * nCoeff);
                return;
        }
}

%extend SMS_AnalParams 
{
        SMS_AnalParams()
        {
                SMS_AnalParams *s = (SMS_AnalParams *)malloc(sizeof(SMS_AnalParams));
                sms_initAnalParams(s);
                return s;
        }
}

%extend SMS_SynthParams 
{
        SMS_SynthParams()
        {
                SMS_SynthParams *s = (SMS_SynthParams *)malloc(sizeof(SMS_SynthParams));
                sms_initSynthParams(s);
                return s;
        }
}

%extend SMS_SpectralPeaks 
{
        SMS_SpectralPeaks(int n)
        {
                SMS_SpectralPeaks *s = (SMS_SpectralPeaks *)malloc(sizeof(SMS_SpectralPeaks));
                s->nPeaks = n;
                if ((s->pSpectralPeaks =
                      (SMS_Peak *)calloc (s->nPeaks, sizeof(SMS_Peak))) == NULL)
                {
                        sms_error("could not allocate memory for spectral peaks");
                        return(NULL);
                }
                s->nPeaksFound = 0;
                return s;
        }
        void getFreq( int sizeArray, float *pArray )
        {
                if(sizeArray < $self->nPeaksFound)
                {
                        sms_error("numpy array not big enough");
                        return;
                }
                int i;
                for (i = 0; i < $self->nPeaksFound; i++)
                        pArray[i] = $self->pSpectralPeaks[i].fFreq;

        }
        void getMag( int sizeArray, float *pArray )
        {
                if(sizeArray < $self->nPeaksFound)
                {
                        sms_error("numpy array not big enough");
                        return;
                }
                int i;
                for (i = 0; i < $self->nPeaksFound; i++)
                        pArray[i] = $self->pSpectralPeaks[i].fMag;
        }
        void getPhase( int sizeArray, float *pArray )
        {
                if(sizeArray < $self->nPeaksFound)
                {
                        sms_error("numpy array not big enough");
                        return;
                }
                int i;
                for (i = 0; i < $self->nPeaksFound; i++)
                        pArray[i] = $self->pSpectralPeaks[i].fPhase;

        }
}

%extend SMS_Data 
{
        void getSinAmp(int sizeArray, float *pArray)
        {
                if(sizeArray < $self->nTracks)
                {
                        sms_error("numpy array not big enough");
                        return;
                }
                int i;
                for (i = 0; i < $self->nTracks; i++)
                        pArray[i] = $self->pFSinAmp[i];
        }
        void getSinFreq(int sizeArray, float *pArray)
        {
                if(sizeArray < $self->nTracks)
                {
                        sms_error("numpy array not big enough");
                        return;
                }
                int i;
                for (i = 0; i < $self->nTracks; i++)
                        pArray[i] = $self->pFSinFreq[i];
        }
        void getSinPhase(int sizeArray, float *pArray)
        {
                if(sizeArray < $self->nTracks)
                {
                        sms_error("numpy array not big enough");
                        return;
                }
                int i;
                for (i = 0; i < $self->nTracks; i++)
                        pArray[i] = $self->pFSinPha[i];
        }
        void getSinEnv(int sizeArray, float *pArray)
        {
                if(sizeArray < $self->nEnvCoeff)
                {
                        sms_error("numpy array not big enough");
                        return;
                }
                int i;
                for (i = 0; i < $self->nEnvCoeff; i++)
                        pArray[i] = $self->pSpecEnv[i];
        }
        void setSinAmp(int sizeArray, float *pArray)
        {
                if(sizeArray < $self->nTracks)
                {
                        sms_error("numpy array not big enough");
                        return;
                }
                int i;
                for (i = 0; i < $self->nTracks; i++)
                        $self->pFSinAmp[i] = pArray[i];
                                
        }
        void setSinFreq(int sizeArray, float *pArray)
        {
                if(sizeArray < $self->nTracks)
                {
                        sms_error("numpy array not big enough");
                        return;
                }
                int i;
                for (i = 0; i < $self->nTracks; i++)
                        $self->pFSinFreq[i] = pArray[i];
        }
        void setSinPha(int sizeArray, float *pArray)
        {
                if(sizeArray < $self->nTracks)
                {
                        sms_error("numpy array not big enough");
                        return;
                }
                int i;
                for (i = 0; i < $self->nTracks; i++)
                        $self->pFSinPha[i] = pArray[i];
        }
}

%extend SMS_ModifyParams
{
        /* no need to return an error code, if sms_error is called, it will throw an exception in python */
        void setSinEnv(int sizeArray, float *pArray)
        {
                if(!$self->ready)
                {
                        sms_error("modify parameter structure has not been initialized");
                        return;
                }
                if(sizeArray != $self->sizeSinEnv)
                {
                        sms_error("numpy array is not equal to envelope size");
                        return;
                }
                memcpy($self->sinEnv, pArray, sizeof(sfloat) * $self->sizeSinEnv);
        }
}
