#include "test_partial_tracking.h"

using namespace simpl;

// ---------------------------------------------------------------------------
//	TestMQPartialTracking
// ---------------------------------------------------------------------------
void TestMQPartialTracking::setUp() {
    _sf = SndfileHandle(TEST_AUDIO_FILE);

    if(_sf.error() > 0) {
        throw Exception(std::string("Could not open audio file: ") +
                        std::string(TEST_AUDIO_FILE));
    }
}

void TestMQPartialTracking::test_basic() {
    int hop_size = 256;
    int frame_size = 2048;
    int num_samples = 4096;

    _pd.clear();
    _pt.reset();

    _pd.hop_size(hop_size);
    _pd.frame_size(frame_size);

    std::vector<sample> audio(_sf.frames(), 0.0);
    _sf.read(&audio[0], (int)_sf.frames());

    Frames frames = _pd.find_peaks(num_samples,
                                   &(audio[(int)_sf.frames() / 2]));
    frames = _pt.find_partials(frames);

    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
    }
}

void TestMQPartialTracking::test_peaks() {
    int num_frames = 8;
    Frames frames;

    _pd.clear();
    _pt.reset();

    for(int i = 0; i < num_frames; i++) {
        Frame* f = new Frame();
        f->add_peak(0.4, 220, 0, 0);
        f->add_peak(0.2, 440, 0, 0);
        frames.push_back(f);
    }

    _pt.find_partials(frames);
    for(int i = 0; i < num_frames; i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.4, frames[i]->partial(0)->amplitude,
                                     PRECISION);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(220, frames[i]->partial(0)->frequency,
                                     PRECISION);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.2, frames[i]->partial(1)->amplitude,
                                     PRECISION);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(440, frames[i]->partial(1)->frequency,
                                     PRECISION);
    }

    for(int i = 0; i < num_frames; i++) {
        delete frames[i];
    }
}


// ---------------------------------------------------------------------------
//	TestSMSPartialTracking
// ---------------------------------------------------------------------------
void TestSMSPartialTracking::setUp() {
    _sf = SndfileHandle(TEST_AUDIO_FILE);

    if(_sf.error() > 0) {
        throw Exception(std::string("Could not open audio file: ") +
                        std::string(TEST_AUDIO_FILE));
    }

    _pt.realtime(true);
    _pt.max_frame_delay(2);
    _pt.max_partials(5);
}

void TestSMSPartialTracking::test_basic() {
    int hop_size = 256;
    int frame_size = 2048;
    int num_samples = 4096;

    _pd.clear();
    _pt.reset();

    _pd.hop_size(hop_size);
    _pd.frame_size(frame_size);

    std::vector<sample> audio(_sf.frames(), 0.0);
    _sf.read(&audio[0], (int)_sf.frames());

    Frames frames = _pd.find_peaks(num_samples,
                                   &(audio[(int)_sf.frames() / 2]));
    frames = _pt.find_partials(frames);

    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
    }
}

void TestSMSPartialTracking::test_change_num_partials() {
    int hop_size = 256;
    int frame_size = 2048;
    int num_samples = 4096;
    int max_partials = 10;

    _pd.clear();
    _pt.reset();

    _pd.hop_size(hop_size);
    _pd.frame_size(frame_size);

    _pd.max_peaks(max_partials);
    _pt.max_partials(max_partials);

    std::vector<sample> audio(_sf.frames(), 0.0);
    _sf.read(&audio[0], (int)_sf.frames());

    Frames frames = _pd.find_peaks(num_samples,
                                   &(audio[(int)_sf.frames() / 2]));
    frames = _pt.find_partials(frames);

    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() == max_partials);
        CPPUNIT_ASSERT(frames[i]->num_partials() == max_partials);
    }
}

void TestSMSPartialTracking::test_peaks() {
    int num_frames = 8;
    Frames frames;

    _pd.clear();
    _pt.reset();

    for(int i = 0; i < num_frames; i++) {
        Frame* f = new Frame();
        f->add_peak(0.4, 220, 0, 0);
        f->add_peak(0.2, 440, 0, 0);
        frames.push_back(f);
    }

    _pt.find_partials(frames);
    for(int i = 1; i < num_frames; i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.4, frames[i]->partial(0)->amplitude,
                                     PRECISION);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(220, frames[i]->partial(0)->frequency,
                                     PRECISION);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.2, frames[i]->partial(1)->amplitude,
                                     PRECISION);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(440, frames[i]->partial(1)->frequency,
                                     PRECISION);
    }

    for(int i = 0; i < num_frames; i++) {
        delete frames[i];
    }
}

void TestSMSPartialTracking::test_streaming() {
    int hop_size = 256;
    int frame_size = 2048;
    int num_frames = 10;
    int max_partials = 10;

    _pd.clear();
    _pt.reset();

    _pd.hop_size(hop_size);
    _pd.frame_size(frame_size);

    _pd.max_peaks(max_partials);
    _pt.max_partials(max_partials);

    std::vector<sample> audio(_sf.frames(), 0.0);
    _sf.read(&audio[0], (int)_sf.frames());

    for(int i = 0, n = (int)_sf.frames() / 2; i < num_frames; i++, n += hop_size) {
        Frame f(frame_size, true);
        f.audio(&(audio[n]), frame_size);

        _pd.find_peaks_in_frame(&f);
        _pt.update_partials(&f);

        CPPUNIT_ASSERT(f.num_partials() > 0);
    }
}


// ---------------------------------------------------------------------------
//	TestSndObjPartialTracking
// ---------------------------------------------------------------------------
void TestSndObjPartialTracking::setUp() {
    _sf = SndfileHandle(TEST_AUDIO_FILE);

    if(_sf.error() > 0) {
        throw Exception(std::string("Could not open audio file: ") +
                        std::string(TEST_AUDIO_FILE));
    }

    _pt.max_partials(5);
}

void TestSndObjPartialTracking::test_basic() {
    int hop_size = 256;
    int frame_size = 2048;
    int num_samples = 4096;

    _pd.clear();
    _pt.reset();

    _pd.hop_size(hop_size);
    _pd.frame_size(frame_size);

    std::vector<sample> audio(_sf.frames(), 0.0);
    _sf.read(&audio[0], (int)_sf.frames());

    Frames frames = _pd.find_peaks(num_samples,
                                   &(audio[(int)_sf.frames() / 2]));
    frames = _pt.find_partials(frames);

    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
    }
}

void TestSndObjPartialTracking::test_change_num_partials() {
    int hop_size = 256;
    int frame_size = 2048;
    int num_samples = 4096;
    int max_partials = 10;

    _pd.clear();
    _pt.reset();

    _pd.hop_size(hop_size);
    _pd.frame_size(frame_size);

    _pd.max_peaks(max_partials);
    _pt.max_partials(max_partials);

    std::vector<sample> audio(_sf.frames(), 0.0);
    _sf.read(&audio[0], (int)_sf.frames());

    Frames frames = _pd.find_peaks(num_samples,
                                   &(audio[(int)_sf.frames() / 2]));
    frames = _pt.find_partials(frames);

    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() == max_partials);
        CPPUNIT_ASSERT(frames[i]->num_partials() == max_partials);
    }
}

void TestSndObjPartialTracking::test_peaks() {
    int num_frames = 8;
    Frames frames;

    _pd.clear();
    _pt.reset();

    for(int i = 0; i < num_frames; i++) {
        Frame* f = new Frame();
        f->add_peak(0.4, 220, 0, 0);
        f->add_peak(0.2, 440, 0, 0);
        frames.push_back(f);
    }

    _pt.find_partials(frames);
    for(int i = 1; i < num_frames; i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.4, frames[i]->partial(0)->amplitude,
                                     PRECISION);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(220, frames[i]->partial(0)->frequency,
                                     PRECISION);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.2, frames[i]->partial(1)->amplitude,
                                     PRECISION);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(440, frames[i]->partial(1)->frequency,
                                     PRECISION);
    }

    for(int i = 0; i < num_frames; i++) {
        delete frames[i];
    }
}

void TestSndObjPartialTracking::test_streaming() {
    int hop_size = 256;
    int frame_size = 2048;
    int num_frames = 10;
    int max_partials = 10;

    _pd.clear();
    _pt.reset();

    _pd.hop_size(hop_size);
    _pd.frame_size(frame_size);

    _pd.max_peaks(max_partials);
    _pt.max_partials(max_partials);

    std::vector<sample> audio(_sf.frames(), 0.0);
    _sf.read(&audio[0], (int)_sf.frames());

    for(int i = 0, n = (int)_sf.frames() / 2; i < num_frames; i++, n += hop_size) {
        Frame f(frame_size, true);
        f.audio(&(audio[n]), frame_size);

        _pd.find_peaks_in_frame(&f);
        _pt.update_partials(&f);

        CPPUNIT_ASSERT(f.num_partials() > 0);
    }
}


// ---------------------------------------------------------------------------
//	TestLorisPartialTracking
// ---------------------------------------------------------------------------
void TestLorisPartialTracking::setUp() {
    _sf = SndfileHandle(TEST_AUDIO_FILE);

    if(_sf.error() > 0) {
        throw Exception(std::string("Could not open audio file: ") +
                        std::string(TEST_AUDIO_FILE));
    }
}

void TestLorisPartialTracking::test_basic() {
    int hop_size = 256;
    int frame_size = 2048;
    int num_samples = 4096;

    _pd.clear();
    _pt.reset();

    _pd.hop_size(hop_size);
    _pd.frame_size(frame_size);

    std::vector<sample> audio(_sf.frames(), 0.0);
    _sf.read(&audio[0], (int)_sf.frames());

    Frames frames = _pd.find_peaks(num_samples,
                                   &(audio[(int)_sf.frames() / 2]));
    frames = _pt.find_partials(frames);

    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
    }
}

void TestLorisPartialTracking::test_peaks() {
    int num_frames = 8;
    Frames frames;

    _pd.clear();
    _pt.reset();

    for(int i = 0; i < num_frames; i++) {
        Frame* f = new Frame();
        f->add_peak(0.4, 220, 0, 0);
        f->add_peak(0.2, 440, 0, 0);
        frames.push_back(f);
    }

    _pt.find_partials(frames);
    for(int i = 0; i < num_frames; i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.4, frames[i]->partial(0)->amplitude,
                                     PRECISION);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(220, frames[i]->partial(0)->frequency,
                                     PRECISION);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.2, frames[i]->partial(1)->amplitude,
                                     PRECISION);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(440, frames[i]->partial(1)->frequency,
                                     PRECISION);
    }

    for(int i = 0; i < num_frames; i++) {
        delete frames[i];
    }
}
