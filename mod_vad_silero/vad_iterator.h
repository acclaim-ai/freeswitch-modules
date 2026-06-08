#ifndef VAD_ITERATOR_H
#define VAD_ITERATOR_H

#include <vector>
#include <sstream>
#include <cstring>
#include <limits>
#include <chrono>
#include <iomanip>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <cstdarg>
#include <cmath>    // for std::rint
#if __cplusplus < 201703L
#include <memory>
#endif

#include "onnxruntime_cxx_api.h"
#include "switch.h"

#include "timestamp.h"

// VadIterator class: uses ONNX Runtime to detect speech segments.
class VadIterator {
private:
    // ONNX Runtime resources. The Env and the model Session are process-global and
    // shared across every VadIterator instance (loaded once in global_init), so the
    // model is read from disk and the inference graph built exactly once instead of
    // once per call. Ort::Session::Run is thread-safe, and each call's mutable state
    // lives in the per-instance _state / _context tensors below, so concurrent calls
    // on the shared session are safe. Both are held by shared_ptr so an in-flight
    // call keeps them alive even if global_cleanup() runs at module shutdown
    // (env is declared first so it outlives session on destruction).
    std::shared_ptr<Ort::Env> env;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

    // ----- Context-related additions -----
    const int context_samples = 64;  // For 16kHz, 64 samples are added as context.
    std::vector<float> _context;     // Holds the last 64 samples from the previous chunk (initialized to zero).

    // Original window size (e.g., 32ms corresponds to 512 samples)
    int window_size_samples;
    // Effective window size = window_size_samples + context_samples
    int effective_window_size;

    // Additional declaration: samples per millisecond
    int sr_per_ms;

    // ONNX Runtime input/output buffers
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char*> input_node_names = { "input", "state", "sr" };
    std::vector<float> input;
    unsigned int size_state = 2 * 1 * 128;
    std::vector<float> _state;
    std::vector<int64_t> sr;
    int64_t input_node_dims[2] = {};
    const int64_t state_node_dims[3] = { 2, 1, 128 };
    const int64_t sr_node_dims[1] = { 1 };
    std::vector<Ort::Value> ort_outputs;
    std::vector<const char*> output_node_names = { "output", "stateN" };

    // Model configuration parameters
    int sample_rate;
    float threshold;
    int min_silence_samples;
    int min_silence_samples_at_max_speech;
    int min_speech_samples;
    float max_speech_samples;
    int speech_pad_samples;
    int audio_length_samples;

    // State management
    bool triggered = false;
    unsigned int temp_end = 0;
    unsigned int current_sample = 0;
    int prev_end;
    int next_start = 0;
    timestamp_t current_speech;
    float last_speech_probability;
    bool speech_started_this_frame = false;
    bool speech_ended_this_frame = false;

    // Attach this iterator to the process-global ONNX Env + Session (loaded once in
    // global_init). Throws if the shared session is unavailable so the caller
    // (SileroVADContext::initialize) can fail cleanly. The model path is fixed at
    // global_init time; the argument is kept only for signature compatibility.
    void init_onnx_model(const std::string& model_path) {
        env = get_shared_env();
        session = get_shared_session();
        if (!env || !session) {
            throw std::runtime_error("Silero VAD ONNX session not initialized; call global_init() first");
        }
        (void) model_path;
    }

    // Resets internal state (_state, _context, etc.)
    void reset_states();

    // Inference: runs inference on one chunk of input data.
    // data_chunk is expected to have window_size_samples samples.
    void predict(const std::vector<float>& data_chunk) ;

public:
    // Process the entire audio input (batch mode - resets state)
    void process(const std::vector<float>& input_wav);

    // Process audio chunk for streaming (doesn't reset state)
    void process_chunk(const std::vector<float>& input_chunk);

    // Get the current speech segment if one is in progress
    timestamp_t get_current_speech() const {
        return current_speech;
    }

    // Get the sample rate
    int get_sample_rate() const {
        return sample_rate;
    }

    // Check if speech is currently active (for streaming applications)
    bool is_speech_active() const {
        return triggered;
    }

    // Get the current speech probability (for debugging/monitoring)
    float get_last_speech_probability() const {
        return last_speech_probability;
    }

    // Public method to reset the internal state.
    void reset() {
        reset_states();
    }

    // Process-wide model lifecycle. global_init() loads the shared Ort::Env +
    // Session once (idempotent, thread-safe); global_cleanup() releases them at
    // module shutdown. The get_shared_* accessors return the loaded resources
    // (or null if global_init has not run / failed).
    static int global_init();
    static void global_cleanup();
    static std::shared_ptr<Ort::Env> get_shared_env();
    static std::shared_ptr<Ort::Session> get_shared_session();

public:
    // Constructor: sets model path, sample rate, window size (ms), and other parameters.
    // The parameters are set to match the Python version.
    VadIterator(const std::string ModelPath,
        int Sample_rate = 16000, int windows_frame_size = 32,
        float Threshold = 0.5, int min_silence_duration_ms = 100,
        int speech_pad_ms = 30, int min_speech_duration_ms = 250,
        float max_speech_duration_s = std::numeric_limits<float>::infinity()) ;

    // Destructor
    ~VadIterator();

    // Get speech state transition info for streaming
    struct SpeechTransition {
        bool speech_started;    // true if speech just started
        bool speech_ended;      // true if speech just ended
        float probability;      // current speech probability
    };
    
    SpeechTransition get_speech_transition() const {
        return {speech_started_this_frame, speech_ended_this_frame, last_speech_probability};
    }
};
 
#endif
