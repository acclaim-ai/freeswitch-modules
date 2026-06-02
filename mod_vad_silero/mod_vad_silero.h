#ifndef __MOD_VAD_SILEO_H__
#define __MOD_VAD_SILEO_H__

#include <switch.h>
#include <speex/speex_resampler.h>


#define MY_BUG_NAME "vad_silero"
#define VAD_EVENT_SILERO "vad_silero:detect"

typedef struct private_data
{
  switch_mutex_t *mutex;
  SpeexResamplerState *resampler;

  char *bugname;
  char *strategy;
  char *sessionId;

  int stopping;
  int cleanup_done;  // Flag to track cleanup state
  int sample_rate;
  float threshold;
  int silence_ms;
  int voice_ms;
  int min_speech_ms;  // Store minimum speech duration for timestamp calculation
  float probability;  // Store current speech probability

  void *pVadContext;  // Points to std::shared_ptr<SileroVADContext>* (reference-counted)

  switch_vad_state_t previous_vad_state;  // Track previous state to avoid duplicates
} private_t;

#endif