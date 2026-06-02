#include "mod_vad_silero.h"
#include "silero_vad_wrapper.h"
#include <sys/time.h>

SWITCH_MODULE_SHUTDOWN_FUNCTION(mod_vad_silero_shutdown);
SWITCH_MODULE_LOAD_FUNCTION(mod_vad_silero_load);
SWITCH_MODULE_DEFINITION(mod_vad_silero, mod_vad_silero_load, mod_vad_silero_shutdown, NULL);

static void create_vad_json_payload(switch_vad_state_t state, uint32_t min_speech_ms, uint32_t silence_ms, float probability, char* json_buffer, size_t buffer_size) {
  struct timeval tv;
  long long timestamp_ms;
  const char* event_type;
  
  // Calculate timestamp in milliseconds since epoch
  gettimeofday(&tv, NULL);
  timestamp_ms = (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
  
  // Adjust timestamp based on VAD detection delays
  if (state == SWITCH_VAD_STATE_START_TALKING) {
    // For speech-started, subtract minimum speech duration to get actual start time
    timestamp_ms -= min_speech_ms;
  } else if (state == SWITCH_VAD_STATE_STOP_TALKING) {
    // For speech-stopped, subtract silence duration to get actual end time
    // (speech actually ended when silence began, not when we detected it)
    timestamp_ms -= silence_ms;
  }
  
  // Create JSON payload with probability
  event_type = (state == SWITCH_VAD_STATE_START_TALKING) ? "speech-started" : "speech-stopped";
  snprintf(json_buffer, buffer_size, "{\"event\":\"%s\",\"timestamp\":%lld,\"probability\":%.3f,\"vendor\":\"silero\"}", event_type, timestamp_ms, probability);
}

static void responseHandler(switch_core_session_t* session, switch_vad_state_t state, const char* bugname, const char* json) {
  switch_event_t *event;
  switch_channel_t *channel = switch_core_session_get_channel(session);
    
  switch_event_create_subclass(&event, SWITCH_EVENT_CUSTOM, VAD_EVENT_SILERO);
  switch_channel_event_set_data(channel, event);
  switch_event_add_header_string(event, SWITCH_STACK_BOTTOM, "detected-event", state == SWITCH_VAD_STATE_START_TALKING ? "start_talking" : "stop_talking");
  if (bugname) switch_event_add_header_string(event, SWITCH_STACK_BOTTOM, "media-bugname", bugname);
  if (json) switch_event_add_body(event, "%s", json);
  switch_event_fire(&event);
}

static switch_status_t cleanup_vad_resources(switch_core_session_t *session, char* bugname, int channelIsClosing) {
  switch_channel_t *channel = switch_core_session_get_channel(session);
  switch_media_bug_t *bug = switch_channel_get_private(channel, bugname);
  private_t* userData = NULL;

  if (!bug) {
    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG,
      "cleanup_vad_resources: no bug %s - already cleaned up\n", bugname);
    return SWITCH_STATUS_FALSE;
  }

  userData = (private_t*) switch_core_media_bug_get_user_data(bug);
  if (!userData) {
    return SWITCH_STATUS_FALSE;
  }

  switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "cleanup_vad_resources\n");

  switch_mutex_lock(userData->mutex);

  // Check if cleanup already done
  if (userData->cleanup_done) {
    switch_mutex_unlock(userData->mutex);
    return SWITCH_STATUS_SUCCESS;
  }

  // Set cleanup flag to prevent double cleanup
  userData->cleanup_done = 1;

  // Get the bug again under lock (double-check pattern)
  {
    switch_media_bug_t *bug = switch_channel_get_private(channel, bugname);
    if (bug) {
      switch_channel_set_private(channel, bugname, NULL);
      if (!channelIsClosing) {
        // Only remove bug if we're initiating the stop (not if framework is closing)
        switch_mutex_unlock(userData->mutex);
        switch_core_media_bug_remove(session, &bug);
        switch_mutex_lock(userData->mutex);
      }
    }
  }

  // Clean up allocated strings
  if (userData->bugname) {
    free(userData->bugname);
    userData->bugname = NULL;
  }
  if (userData->strategy) {
    free(userData->strategy);
    userData->strategy = NULL;
  }
  if (userData->sessionId) {
    free(userData->sessionId);
    userData->sessionId = NULL;
  }

  // Clean up VAD context (shared_ptr reference-counted)
  if (userData->pVadContext) {
    silero_vad_destroy(userData->pVadContext);
    userData->pVadContext = NULL;
  }

  // Clean up resampler
  if (userData->resampler) {
    speex_resampler_destroy(userData->resampler);
    userData->resampler = NULL;
  }

  // Reset other fields
  userData->stopping = 0;
  userData->sample_rate = 0;
  userData->threshold = 0.0f;
  userData->silence_ms = 0;
  userData->voice_ms = 0;
  userData->min_speech_ms = 0;
  userData->probability = 0.0f;
  userData->previous_vad_state = SWITCH_VAD_STATE_NONE;

  switch_mutex_unlock(userData->mutex);
  switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO, "cleanup_vad_resources: done\n");

  return SWITCH_STATUS_SUCCESS;
}

static switch_status_t do_stop(switch_core_session_t *session, char* bugname)
{
  switch_channel_t *channel = switch_core_session_get_channel(session);
  switch_media_bug_t *bug = switch_channel_get_private(channel, bugname);
  switch_status_t status = SWITCH_STATUS_SUCCESS;
  private_t* userData = NULL;
  char json[256];
  struct timeval tv;
  long long timestamp_ms;

  if (!bug) {
    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG,
      "do_stop: no bug %s\n", bugname);
    return SWITCH_STATUS_FALSE;
  }

  // Get user data
  userData = (private_t*) switch_core_media_bug_get_user_data(bug);
  if (!userData) {
    return SWITCH_STATUS_FALSE;
  }

  switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO, "do_stop for bug %s\n", bugname);

  switch_mutex_lock(userData->mutex);

  if (!userData->cleanup_done) {
    // Set stopping flag to prevent further processing
    userData->stopping = 1;

    // Check if speech is currently active and send immediate stop event
    if (userData->previous_vad_state == SWITCH_VAD_STATE_START_TALKING) {
      switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO,
        "VAD session stopping while speech is active - sending immediate speech stopped event\n");

      // Create JSON payload for immediate speech stop (no silence duration subtraction)
      // Calculate timestamp in milliseconds since epoch
      gettimeofday(&tv, NULL);
      timestamp_ms = (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;

      // For immediate stop, don't subtract silence duration
      snprintf(json, sizeof(json), "{\"event\":\"speech-stopped\",\"timestamp\":%lld,\"probability\":%.3f,\"vendor\":\"silero\"}", timestamp_ms, userData->probability);

      // Fire the event
      responseHandler(session, SWITCH_VAD_STATE_STOP_TALKING, userData->bugname, json);

      // Update state to reflect that speech has stopped
      userData->previous_vad_state = SWITCH_VAD_STATE_STOP_TALKING;
    }
  }

  switch_mutex_unlock(userData->mutex);

  // Call cleanup with channelIsClosing=0 (we're initiating the stop)
  status = cleanup_vad_resources(session, bugname, 0);

  return status;
}

static switch_bool_t capture_callback(switch_media_bug_t *bug, void *user_data, switch_abc_type_t type)
{
  switch_core_session_t *session = switch_core_media_bug_get_session(bug);
  private_t* userData = (private_t*) user_data;

  switch (type) {
  case SWITCH_ABC_TYPE_INIT:
      switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "Got SWITCH_ABC_TYPE_INIT.\n");
    break;

  case SWITCH_ABC_TYPE_CLOSE:
    {
      char json[256];
      struct timeval tv;
      long long timestamp_ms;
      int is_speech_active = 0;

      switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "Got SWITCH_ABC_TYPE_CLOSE.\n");

      // Set stopping flag to prevent further processing
      if (userData && !userData->cleanup_done) {
        switch_mutex_lock(userData->mutex);

        userData->stopping = 1;

        // Check if speech is currently active and send immediate stop event
        // We check both the previous_vad_state and the current VAD context state
        if (userData->pVadContext) {
          // Get current state from VAD context
          is_speech_active = silero_vad_is_speech_active(userData->pVadContext);

          switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG,
            "VAD close: previous_vad_state=%d, is_speech_active=%d\n",
            userData->previous_vad_state, is_speech_active);

          if (userData->previous_vad_state == SWITCH_VAD_STATE_START_TALKING ||
              is_speech_active == 1) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO,
              "VAD session closing while speech is active - sending immediate speech stopped event\n");

            // Create JSON payload for immediate speech stop (no silence duration subtraction)
            // Calculate timestamp in milliseconds since epoch
            gettimeofday(&tv, NULL);
            timestamp_ms = (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;

            // For immediate stop, don't subtract silence duration
            snprintf(json, sizeof(json), "{\"event\":\"speech-stopped\",\"timestamp\":%lld,\"probability\":%.3f,\"vendor\":\"silero\"}", timestamp_ms, userData->probability);

            // Fire the event
            responseHandler(session, SWITCH_VAD_STATE_STOP_TALKING, userData->bugname, json);

            // Update state to reflect that speech has stopped
            userData->previous_vad_state = SWITCH_VAD_STATE_STOP_TALKING;
          }
        }

        switch_mutex_unlock(userData->mutex);

        // Call cleanup with channelIsClosing=1 (framework is closing the bug)
        cleanup_vad_resources(session, userData->bugname, 1);
      }
    }
    break;
  
  case SWITCH_ABC_TYPE_READ:
    {
      uint8_t data[SWITCH_RECOMMENDED_BUFFER_SIZE];
      switch_frame_t frame;

      if (!userData) return SWITCH_TRUE;

      // Use trylock to avoid blocking the media thread
      if (switch_mutex_trylock(userData->mutex) != SWITCH_STATUS_SUCCESS) {
        return SWITCH_TRUE;
      }

      // Check if we're stopping or already cleaned up
      if (userData->stopping || !userData->pVadContext) {
        switch_mutex_unlock(userData->mutex);
        return SWITCH_TRUE;
      }

      memset(&frame, 0, sizeof(frame));
      frame.data = data;
      frame.buflen = SWITCH_RECOMMENDED_BUFFER_SIZE;

      while (switch_core_media_bug_read(bug, &frame, SWITCH_TRUE) == SWITCH_STATUS_SUCCESS && !switch_test_flag((&frame), SFF_CNG)) {
        if (frame.datalen && userData->pVadContext && !userData->stopping) {
          int vad_state;

          int result = silero_vad_process(userData->pVadContext, userData->resampler, (int16_t*) frame.data, frame.samples, &vad_state);
          if (result == 0) {
            switch_vad_state_t state = (switch_vad_state_t)vad_state;

            // Get current probability from VAD context
            userData->probability = silero_vad_get_probability(userData->pVadContext);

            // Only fire events if state actually changed
            if (state != userData->previous_vad_state) {
              //switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG,
              //  "VAD state change detected: %d -> %d (prob=%.3f)\n",
              //  userData->previous_vad_state, state, userData->probability);

              switch (state)
              {
                case SWITCH_VAD_STATE_START_TALKING:
                case SWITCH_VAD_STATE_STOP_TALKING:
                  {
                    // Create JSON payload using helper function
                    char json[256];
                    create_vad_json_payload(state, userData->min_speech_ms, userData->silence_ms, userData->probability, json, sizeof(json));
                    responseHandler(session, state, userData->bugname, json);
                  }
                  // Add NULL check before strcasecmp to prevent segfault
                  if (state == SWITCH_VAD_STATE_STOP_TALKING && userData->strategy && !strcasecmp(userData->strategy, "one-shot")) {
                    userData->stopping = 1;
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG,
                      "One-shot VAD detection completed, stopping detection\n");
                    // Unlock before calling do_stop to avoid deadlock
                    switch_mutex_unlock(userData->mutex);
                    // Stop VAD detection directly - do_stop will handle cleanup
                    do_stop(session, userData->bugname);
                    return SWITCH_TRUE;
                  }
                break;

                case SWITCH_VAD_STATE_TALKING:
                case SWITCH_VAD_STATE_NONE:
                default:
                  break;
              }
              userData->previous_vad_state = state;  // Update previous state
            }
          }
        }
      }

      switch_mutex_unlock(userData->mutex);
    }
    break;

  case SWITCH_ABC_TYPE_WRITE:
  default:
    break;
  }

  return SWITCH_TRUE;
}

static switch_status_t start_capture(switch_core_session_t *session, switch_media_bug_flag_t flags, 
  char* action, float threshold, uint32_t silence_ms, uint32_t speech_pad_ms, uint32_t min_speech_ms, char* bugname)
{
  switch_channel_t *channel = switch_core_session_get_channel(session);
  const char* szSessionId = switch_core_session_get_uuid(session);
  switch_media_bug_t *bug;
  switch_status_t status = SWITCH_STATUS_SUCCESS; 
  private_t* userData = (private_t *) switch_core_session_alloc(session, sizeof(*userData));
  switch_codec_implementation_t read_impl = { 0 };
  uint32_t samples_per_second;
  int err;

  if (switch_channel_get_private(channel, bugname)) {
    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "removing bug from previous vad detection\n");
    do_stop(session, bugname);
  }

  if (switch_channel_pre_answer(channel) != SWITCH_STATUS_SUCCESS) {
    return SWITCH_STATUS_FALSE;
  }
  switch_core_session_get_read_impl(session, &read_impl);
  samples_per_second = !strcasecmp(read_impl.iananame, "g722") ? read_impl.actual_samples_per_second : read_impl.samples_per_second;

  userData->stopping = 0;
  userData->cleanup_done = 0;  // Initialize cleanup flag
  userData->sample_rate = samples_per_second;
  userData->threshold = threshold;
  userData->silence_ms = silence_ms;
  userData->voice_ms = 0;
  userData->min_speech_ms = min_speech_ms;
  userData->probability = 0.0f;  // Initialize probability
  userData->pVadContext = NULL;
  userData->previous_vad_state = SWITCH_VAD_STATE_NONE;

  // Initialize mutex
  switch_mutex_init(&userData->mutex, SWITCH_MUTEX_NESTED, switch_core_session_get_pool(session));

  if (samples_per_second != 16000) {
    userData->resampler = speex_resampler_init(1, samples_per_second, 16000, SWITCH_RESAMPLE_QUALITY, &err);
    if (0 != err) {
      switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "Failed to initialize resampler\n");
      return SWITCH_STATUS_FALSE;
    }
  }
  else {
    userData->resampler = NULL;
  }

  // Create per-session Silero VAD context with shared_ptr (window size hardcoded to 32ms)
  userData->pVadContext = silero_vad_create(szSessionId, userData->sample_rate, threshold, silence_ms, speech_pad_ms, min_speech_ms);
  if (userData->pVadContext) {
    userData->bugname = strdup(bugname);
    userData->strategy = strdup(action);
    userData->sessionId = strdup(switch_core_session_get_uuid(session));
    
    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, 
      "configured Silero VAD incoming sample_rate: %d, threshold: %.2f silence_ms: %d speech_pad_ms: %d min_speech_ms: %d (window: 32ms)\n", 
      userData->sample_rate, threshold, silence_ms, speech_pad_ms, min_speech_ms);
      
    if ((status = switch_core_media_bug_add(session, bugname, NULL, capture_callback, userData, 0, flags, &bug)) != SWITCH_STATUS_SUCCESS) {
      switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "Failed to initiate vad resource\n");
      return status;
    }
    switch_channel_set_private(channel, bugname, bug);
  } else {
    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "Failed to create Silero VAD context\n");
    return SWITCH_STATUS_FALSE;
  }
  return SWITCH_STATUS_SUCCESS;
}

#define VAD_API_SYNTAX "<uuid> [start|stop] [one-shot|continuous] threshold silence-ms speech-pad-ms min-speech-ms [bugname] (use 0 for defaults: silence=100, padding=30, speech=250)"
SWITCH_STANDARD_API(vad_silero_function) {
  char *mycmd = NULL, *argv[8] = { 0 };
  int argc = 0;
  switch_status_t status = SWITCH_STATUS_FALSE;
  switch_media_bug_flag_t flags = SMBF_READ_STREAM /* | SMBF_WRITE_STREAM | SMBF_READ_PING */;

  if (!zstr(cmd) && (mycmd = strdup(cmd))) {
    argc = switch_separate_string(mycmd, ' ', argv, (sizeof(argv) / sizeof(argv[0])));
  }

  if (zstr(cmd) || 
      (!strcasecmp(argv[1], "stop") && argc < 2) ||
      (!strcasecmp(argv[1], "start") && argc < 6) ||
      zstr(argv[0])) {
    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "Error with command %s %s %s.\n", cmd, argv[0], argv[1]);
    stream->write_function(stream, "-USAGE: %s\n", VAD_API_SYNTAX);
    goto done;
  } else {
    switch_core_session_t *lsession = NULL;

    if ((lsession = switch_core_session_locate(argv[0]))) {
      if (!strcasecmp(argv[1], "stop")) {
        char *bugname = argc > 2 ? argv[2] : MY_BUG_NAME;
        status = do_stop(lsession, bugname);
      } else if (!strcasecmp(argv[1], "start")) {
        char* action = argv[2];
        float threshold = atof(argv[3]);
        uint32_t silence_ms = atoi(argv[4]);
        uint32_t speech_pad_ms = atoi(argv[5]);
        uint32_t min_speech_ms = atoi(argv[6]);
        char *bugname = argc > 7 ? argv[7] : MY_BUG_NAME;

        // Use defaults if parameters are set to 0
        if (silence_ms == 0) silence_ms = 100;      // Default: 100ms
        if (speech_pad_ms == 0) speech_pad_ms = 30;  // Default: 30ms  
        if (min_speech_ms == 0) min_speech_ms = 250; // Default: 250ms

        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO, 
          "start vad detect action: %s threshold: %.2f silence-ms: %d speech-pad-ms: %d min-speech-ms: %d bugname: %s.\n",
          action, threshold, silence_ms, speech_pad_ms, min_speech_ms, bugname);
        status = start_capture(lsession, flags, action, threshold, silence_ms, speech_pad_ms, min_speech_ms, bugname);
      }
      switch_core_session_rwunlock(lsession);
    }
  }

  if (status == SWITCH_STATUS_SUCCESS) {
    stream->write_function(stream, "+OK Success\n");
  } else {
    stream->write_function(stream, "-ERR Operation Failed\n");
  }

  done:

  switch_safe_free(mycmd);
  return SWITCH_STATUS_SUCCESS;
}
SWITCH_MODULE_LOAD_FUNCTION(mod_vad_silero_load)
{
  switch_api_interface_t *api_interface;

  /* create/register custom event message type */
  if (switch_event_reserve_subclass(VAD_EVENT_SILERO) != SWITCH_STATUS_SUCCESS) {
    switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR, "Couldn't register subclass %s!\n", VAD_EVENT_SILERO);
    return SWITCH_STATUS_TERM;
  }

  /* Initialize global Silero VAD model */
  if (silero_vad_global_init() != 0) {
    switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR, "Failed to initialize global Silero VAD model\n");
    return SWITCH_STATUS_TERM;
  }

  /* connect my internal structure to the blank pointer passed to me */
  *module_interface = switch_loadable_module_create_module_interface(pool, modname);

  switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_NOTICE, "VAD detection API (Silero) loading..\n");

  switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_NOTICE, "VAD detection API (Silero) successfully loaded\n");

  SWITCH_ADD_API(api_interface, "uuid_vad_silero", "VAD detection API (Silero)", vad_silero_function, VAD_API_SYNTAX);
  switch_console_set_complete("add uuid_vad_silero start [one-shot|continuous] threshold silence-ms speech-pad-ms min-speech-ms [bugname]");
  switch_console_set_complete("add uuid_vad_silero stop [bugname]");

  /* indicate that the module should continue to be loaded */
  return SWITCH_STATUS_SUCCESS;
}

/*
  Called when the system shuts down
  Macro expands to: switch_status_t mod_vad_silero_shutdown() */
SWITCH_MODULE_SHUTDOWN_FUNCTION(mod_vad_silero_shutdown)
{
  switch_event_free_subclass(VAD_EVENT_SILERO);
  return SWITCH_STATUS_SUCCESS;
}