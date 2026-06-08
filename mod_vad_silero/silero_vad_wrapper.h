#ifndef SILERO_VAD_WRAPPER_H
#define SILERO_VAD_WRAPPER_H

#include <speex/speex_resampler.h>

#ifdef __cplusplus
extern "C" {
#endif

// Global model management functions
/**
 * Initialize the global Silero VAD model (called once when module loads)
 * @return 0 on success, -1 on error
 */
int silero_vad_global_init(void);

/**
 * Cleanup the global Silero VAD model (called when module unloads)
 */
void silero_vad_global_cleanup(void);

// C wrapper functions for Silero VAD
// These functions manage the C++ SileroVADContext class through void* pointers

/**
 * Create a new Silero VAD context
 * @param sample_rate Input audio sample rate (8000 or 16000)
 * @param threshold VAD threshold (0.0 to 1.0)
 * @param min_silence_duration_ms Minimum silence duration in milliseconds
 * @param speech_pad_ms Speech padding in milliseconds
 * @param min_speech_duration_ms Minimum speech duration in milliseconds
 * @return Pointer to VAD context, or NULL on failure
 */
void* silero_vad_create(const char* sessionId, int sample_rate, float threshold, int min_silence_duration_ms, 
                       int speech_pad_ms, int min_speech_duration_ms);

/**
 * Process audio data and get VAD state
 * @param ctx VAD context pointer
 * @param resampler Speex resampler state (can be NULL if no resampling needed)
 * @param audio_data Input audio samples (int16_t)
 * @param samples Number of audio samples
 * @param vad_state Output VAD state (SWITCH_VAD_STATE_*)
 * @return 0 on success, -1 on error
 */
int silero_vad_process(void* ctx, SpeexResamplerState *resampler, const int16_t* audio_data, int samples, int* vad_state);

/**
 * Get the current speech probability from the VAD context
 * @param ctx VAD context pointer
 * @return Current speech probability (0.0 to 1.0), or -1.0 on error
 */
float silero_vad_get_probability(void* ctx);

/**
 * Check if speech is currently active in the VAD context
 * @param ctx VAD context pointer
 * @return 1 if speech is active, 0 if not, -1 on error
 */
int silero_vad_is_speech_active(void* ctx);

/**
 * Destroy and cleanup VAD context
 * @param ctx VAD context pointer to destroy
 */
void silero_vad_destroy(void* ctx);

#ifdef __cplusplus
}
#endif

#endif // SILERO_VAD_WRAPPER_H 
