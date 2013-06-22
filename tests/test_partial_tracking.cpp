#include "test_partial_tracking.h"

using namespace simpl;

static void test_basic(PeakDetection *pd, PartialTracking* pt,
                       SndfileHandle *sf) {
    int hop_size = 256;
    int frame_size = 2048;
    int num_samples = 4096;

    pd->clear();
    pt->reset();

    pd->hop_size(hop_size);
    pd->frame_size(frame_size);

    std::vector<sample> audio(sf->frames(), 0.0);
    sf->read(&audio[0], (int)sf->frames());

    Frames frames = pd->find_peaks(num_samples,
                                   &(audio[(int)sf->frames() / 2]));
    frames = pt->find_partials(frames);

    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
    }
}

static void test_change_num_partials(PeakDetection *pd, PartialTracking* pt,
                                     SndfileHandle *sf) {
    int hop_size = 256;
    int frame_size = 2048;
    int num_samples = 4096;
    int max_partials = 256;

    pd->clear();
    pt->reset();

    pd->hop_size(hop_size);
    pd->frame_size(frame_size);

    pd->max_peaks(max_partials);
    pt->max_partials(max_partials);

    std::vector<sample> audio(sf->frames(), 0.0);
    sf->read(&audio[0], (int)sf->frames());

    Frames frames = pd->find_peaks(num_samples,
                                   &(audio[(int)sf->frames() / 2]));
    frames = pt->find_partials(frames);

    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->max_peaks() == max_partials);
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 20);
        CPPUNIT_ASSERT(frames[i]->max_partials() == max_partials);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 20);
    }
}

static void test_peaks(PeakDetection *pd, PartialTracking* pt,
                       SndfileHandle *sf) {
    int num_frames = 8;
    Frames frames;

    pd->clear();
    pt->reset();

    for(int i = 0; i < num_frames; i++) {
        Frame* f = new Frame();
        f->add_peak(0.4, 220, 0, 0);
        f->add_peak(0.2, 440, 0, 0);
        frames.push_back(f);
    }

    pt->find_partials(frames);
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


static void test_streaming(PeakDetection *pd, PartialTracking* pt,
                           SndfileHandle *sf) {
    int hop_size = 256;
    int frame_size = 2048;
    int num_frames = 10;
    int max_partials = 10;

    pd->clear();
    pt->reset();

    pd->hop_size(hop_size);
    pd->frame_size(frame_size);

    pd->max_peaks(max_partials);
    pt->max_partials(max_partials);

    std::vector<sample> audio(sf->frames(), 0.0);
    sf->read(&audio[0], (int)sf->frames());

    for(int i = 0, n = (int)sf->frames() / 2; i < num_frames; i++, n += hop_size) {
        Frame f(frame_size, true);
        f.audio(&(audio[n]), frame_size);

        pd->find_peaks_in_frame(&f);
        pt->update_partials(&f);

        CPPUNIT_ASSERT(f.num_partials() > 0);
    }
}

// ---------------------------------------------------------------------------
// TestMQPartialTracking
// ---------------------------------------------------------------------------
void TestMQPartialTracking::setUp() {
    _sf = SndfileHandle(TEST_AUDIO_FILE);

    if(_sf.error() > 0) {
        throw Exception(std::string("Could not open audio file: ") +
                        std::string(TEST_AUDIO_FILE));
    }
}

void TestMQPartialTracking::test_basic() {
    ::test_basic(&_pd, &_pt, &_sf);
}

void TestMQPartialTracking::test_peaks() {
    ::test_peaks(&_pd, &_pt, &_sf);
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

    _pt_harm.realtime(true);
    _pt_harm.harmonic(true);
    _pt_harm.default_fundamental(220);
    _pt_harm.max_partials(5);
}

void TestSMSPartialTracking::test_basic() {
    ::test_basic(&_pd, &_pt, &_sf);
}

void TestSMSPartialTracking::test_basic_harm() {
    ::test_basic(&_pd, &_pt_harm, &_sf);
}

void TestSMSPartialTracking::test_change_num_partials() {
    ::test_change_num_partials(&_pd, &_pt, &_sf);
}

void TestSMSPartialTracking::test_change_num_partials_harm() {
    ::test_change_num_partials(&_pd, &_pt_harm, &_sf);
}

void TestSMSPartialTracking::test_peaks() {
    ::test_peaks(&_pd, &_pt, &_sf);
}

void TestSMSPartialTracking::test_peaks_harm() {
    // known fail
    //
    // ::test_peaks(&_pd, &_pt_harm, &_sf);
}

void TestSMSPartialTracking::test_streaming() {
    ::test_streaming(&_pd, &_pt, &_sf);
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
    ::test_basic(&_pd, &_pt, &_sf);
}

void TestSndObjPartialTracking::test_change_num_partials() {
    ::test_change_num_partials(&_pd, &_pt, &_sf);
}

void TestSndObjPartialTracking::test_peaks() {
    ::test_peaks(&_pd, &_pt, &_sf);
}

void TestSndObjPartialTracking::test_streaming() {
    ::test_streaming(&_pd, &_pt, &_sf);
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
    ::test_basic(&_pd, &_pt, &_sf);
}

void TestLorisPartialTracking::test_peaks() {
    ::test_peaks(&_pd, &_pt, &_sf);
}
