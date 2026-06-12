#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>
#include <string>
#include <cstring>
#include <cmath>
#include <mutex>
#include "jambonz/circular_buffer.h"

#include "vad_iterator.h"
#include "mod_vad_silero.h"

#if defined(USE_AVX2)
#include <immintrin.h>
#elif defined(USE_SSE2)
#include <emmintrin.h>
#endif

#define WINDOW_SIZE_16K (512)
#define WINDOW_SIZE_8K (256)

// Vectorized int16_t to float conversion
static void convert_int16_to_float(const int16_t* input, float* output, size_t samples) {
#if defined(USE_AVX2)
    // AVX2 vectorized conversion
    const size_t vector_size = 8; // AVX2 processes 8 floats at a time
    const size_t vector_samples = samples & ~(vector_size - 1); // Round down to multiple of 8
    
    // Load the scaling factor (32768.0f) into a vector
    __m256 scale = _mm256_set1_ps(32768.0f);
    
    // Process 8 samples at a time
    for (size_t i = 0; i < vector_samples; i += vector_size) {
        // Load 8 int16_t values
        __m128i int16_vec = _mm_loadu_si128((__m128i*)(input + i));
        
        // Convert to 32-bit integers (sign extend)
        __m256i int32_vec = _mm256_cvtepi16_epi32(int16_vec);
        
        // Convert to float
        __m256 float_vec = _mm256_cvtepi32_ps(int32_vec);
        
        // Divide by 32768.0f
        float_vec = _mm256_div_ps(float_vec, scale);
        
        // Store result
        _mm256_storeu_ps(output + i, float_vec);
    }
    
    // Handle remaining samples with scalar code
    for (size_t i = vector_samples; i < samples; i++) {
        output[i] = static_cast<float>(input[i]) / 32768.0f;
    }
#elif defined(USE_SSE2)
    // SSE2 vectorized conversion
    const size_t vector_size = 4; // SSE2 processes 4 floats at a time
    const size_t vector_samples = samples & ~(vector_size - 1); // Round down to multiple of 4
    
    // Load the scaling factor (32768.0f) into a vector
    __m128 scale = _mm_set1_ps(32768.0f);
    
    // Process 4 samples at a time
    for (size_t i = 0; i < vector_samples; i += vector_size) {
        // Load 4 int16_t values (64 bits)
        __m128i int16_vec = _mm_loadl_epi64((__m128i*)(input + i));
        
        // Convert to 32-bit integers (sign extend; _mm_cvtepi16_epi32 is SSE4.1)
        __m128i sign_vec = _mm_srai_epi16(int16_vec, 15);
        __m128i int32_vec = _mm_unpacklo_epi16(int16_vec, sign_vec);
        
        // Convert to float
        __m128 float_vec = _mm_cvtepi32_ps(int32_vec);
        
        // Divide by 32768.0f
        float_vec = _mm_div_ps(float_vec, scale);
        
        // Store result
        _mm_storeu_ps(output + i, float_vec);
    }
    
    // Handle remaining samples with scalar code
    for (size_t i = vector_samples; i < samples; i++) {
        output[i] = static_cast<float>(input[i]) / 32768.0f;
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < samples; i++) {
        output[i] = static_cast<float>(input[i]) / 32768.0f;
    }
#endif
}

// Silero VAD context class using VadIterator
class SileroVADContext {
private:
    CircularBuffer audio_buffer;
    int sample_rate;
    float threshold;
    bool is_initialized;
    int current_vad_state;   // Track current VAD state to avoid redundant events

    // VadIterator configuration parameters
    int min_silence_duration_ms;
    int speech_pad_ms;
    int min_speech_duration_ms;

    // VadIterator instance
    std::unique_ptr<VadIterator> vad_iterator;
    std::string model_path;

    std::string session_id;

    // Thread safety - owned by this object (not tied to FreeSWITCH session lifecycle)
    // Mutable allows locking in const methods like get_probability()
    mutable std::mutex context_mutex;

public:
    SileroVADContext(const char* session_id, int sample_rate, float threshold, int min_silence_duration_ms, 
                    int speech_pad_ms, int min_speech_duration_ms) 
        : audio_buffer(2048), sample_rate(sample_rate), threshold(threshold), 
          is_initialized(false), current_vad_state(SWITCH_VAD_STATE_NONE),
          min_silence_duration_ms(min_silence_duration_ms), speech_pad_ms(speech_pad_ms),
          min_speech_duration_ms(min_speech_duration_ms), session_id(session_id) {
        
        // Get model path from environment or use default
        const char* env_model_path = getenv("SILERO_VAD_MODEL_PATH");
        if (env_model_path) {
            model_path = env_model_path;
        } else {
            model_path = "/usr/local/share/silero_vad/silero_vad.onnx";
        }
    }

    int initialize() {
        if (is_initialized) {
            return 0;
        }

        try {
            // Validate model using static method
            if (VadIterator::global_init() != 0) {
                switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR, 
                    "Silero VAD model validation failed\n");
                return -1;
            }

            // Create VadIterator instance (always use 16kHz since we resample to 16kHz)
            vad_iterator = std::make_unique<VadIterator>(
                model_path,           // model path
                16000,               // sample rate (always 16kHz after resampling)
                32,                  // window frame size (ms)
                threshold,            // threshold
                min_silence_duration_ms,// min silence duration (ms)
                speech_pad_ms,         // speech pad (ms)
                min_speech_duration_ms,// min speech duration (ms)
                std::numeric_limits<float>::infinity()  // max speech duration
            );

            is_initialized = true;
            switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, 
                "Silero VAD context initialized with sample_rate: %d, threshold: %.2f\n",
                sample_rate, threshold);

            return 0;
        } catch (const std::exception& e) {
            switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR, 
                "Error during Silero VAD initialization: %s\n", e.what());
            return -1;
        }
    }

    int process_audio(SpeexResamplerState *resampler, const int16_t* audio_data, int samples, int* vad_state) {
        std::lock_guard<std::mutex> lock(context_mutex);

        // Calculate window size based on 16kHz (32ms window) since we resample to 16kHz
        size_t window_size = (16000 * 32) / 1000;  // 32ms worth of samples at 16kHz = 512
        if (!is_initialized || !vad_iterator) {
            return -1;
        }

        try {
          if (!resampler) {
            audio_buffer.add(audio_data, samples);
          }
          else {
            spx_uint32_t in_len = samples;
						spx_int16_t out[SWITCH_RECOMMENDED_BUFFER_SIZE];
						spx_uint32_t out_len = SWITCH_RECOMMENDED_BUFFER_SIZE;

           speex_resampler_process_interleaved_int(resampler,
              (const spx_int16_t *) audio_data,
              (spx_uint32_t *) &in_len,
              &out[0],
              &out_len);

            audio_buffer.add(&out[0], out_len);
          }

          if (audio_buffer.size() < window_size) {
            return 0;
          }
          // Fixed-size frame buffer (no VLA): 512 samples == 32ms @ 16kHz == window_size.
          int16_t data[WINDOW_SIZE_16K];
          auto count = audio_buffer.getBlock(&data[0], window_size);
          if (count != window_size) {
            return -1;
          }

          // Convert int16_t to float for VadIterator using optimized vectorized conversion
          std::vector<float> float_audio(static_cast<size_t>(window_size));
          convert_int16_to_float(data, float_audio.data(), count);

          // Process the audio chunk through VadIterator (streaming mode)
          vad_iterator->process_chunk(float_audio);

          // Get speech state transition info
          auto transition = vad_iterator->get_speech_transition();

          // Log state transitions and determine new VAD state
          int new_vad_state = transition.speech_started ? SWITCH_VAD_STATE_START_TALKING :
                             transition.speech_ended ? SWITCH_VAD_STATE_STOP_TALKING :
                             current_vad_state;

          if (transition.speech_started || transition.speech_ended) {
              switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO,
                  "%s Silero VAD: Speech %s (prob=%.3f)\n", session_id.c_str(),
                  transition.speech_started ? "started" : "ended", transition.probability);
          }

          // Only emit state change if the state actually changed
          if (new_vad_state != current_vad_state) {
              current_vad_state = new_vad_state;
              *vad_state = new_vad_state;
          } else {
              *vad_state = SWITCH_VAD_STATE_NONE; // No state change
          }

          return 0;

      } catch (const std::exception& e) {
          switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR,
              "%s Error during Silero VAD processing: %s\n", session_id.c_str(), e.what());
          return -1;
      }
    }

    void cleanup() {
        std::lock_guard<std::mutex> lock(context_mutex);

        if (is_initialized) {
            is_initialized = false;
        }
        audio_buffer.clear();
        vad_iterator.reset();
    }

    float get_probability() const {
        std::lock_guard<std::mutex> lock(context_mutex);

        if (vad_iterator) {
            return vad_iterator->get_last_speech_probability();
        }
        return -1.0f;
    }

    bool is_speech_active() const {
        std::lock_guard<std::mutex> lock(context_mutex);

        if (vad_iterator) {
            return vad_iterator->is_speech_active();
        }
        return false;
    }
};

// Helper to get shared_ptr from void* (similar to getAP in mod_deepgram_transcribe)
static std::shared_ptr<SileroVADContext> getVadContext(void* pVadContext) {
    if (pVadContext) {
        auto pCtx = reinterpret_cast<std::shared_ptr<SileroVADContext>*>(pVadContext);
        return *pCtx;
    }
    return nullptr;
}

// C wrapper functions
extern "C" {

int silero_vad_global_init(void) {
    return VadIterator::global_init();
}

void silero_vad_global_cleanup(void) {
    // Release the shared ONNX Env + Session loaded in global_init.
    VadIterator::global_cleanup();
}

void* silero_vad_create(const char* session_id, int sample_rate, float threshold, int min_silence_duration_ms,
                        int speech_pad_ms, int min_speech_duration_ms) {
    auto ctx = std::make_shared<SileroVADContext>(session_id, sample_rate, threshold, min_silence_duration_ms,
                                                   speech_pad_ms, min_speech_duration_ms);
    if (ctx->initialize() != 0) {
        return nullptr;
    }

    // Store the shared_ptr itself (not the raw pointer) - wrapped for C compatibility
    return new std::shared_ptr<SileroVADContext>(ctx);
}

int silero_vad_process(void* ctx, SpeexResamplerState *resampler, const int16_t* audio_data, int samples, int* vad_state) {
    auto vadCtx = getVadContext(ctx);
    if (!vadCtx) return -1;
    return vadCtx->process_audio(resampler, audio_data, samples, vad_state);
}

void silero_vad_destroy(void* ctx) {
    if (ctx) {
        auto vadCtx = getVadContext(ctx);
        if (vadCtx) {
            vadCtx->cleanup();
        }
        // Delete the shared_ptr wrapper allocated in silero_vad_create
        delete reinterpret_cast<std::shared_ptr<SileroVADContext>*>(ctx);
    }
}

float silero_vad_get_probability(void* ctx) {
    auto vadCtx = getVadContext(ctx);
    if (!vadCtx) return -1.0f;
    return vadCtx->get_probability();
}

int silero_vad_is_speech_active(void* ctx) {
    auto vadCtx = getVadContext(ctx);
    if (!vadCtx) return -1;
    return vadCtx->is_speech_active() ? 1 : 0;
}

} // extern "C"

