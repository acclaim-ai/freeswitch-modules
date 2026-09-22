#ifndef __LWS_GLUE_H__
#define __LWS_GLUE_H__

#include "mod_audio_fork.h"

int parse_ws_uri(switch_channel_t *channel, const char* szServerUri, char* host, char *path, unsigned int* pPort, int* pSslFlags);

switch_status_t fork_init();
switch_status_t fork_cleanup();
switch_status_t fork_session_init(switch_core_session_t *session, responseHandler_t responseHandler,
  uint32_t samples_per_second, char *host, unsigned int port, char* path, int sampling, int sslFlags, int channels, 
  char *bugname, char* metadata, int bidirectional_audio_enable,
  int bidirectional_audio_stream, int bidirectional_audio_sample_rate, const char* ws_codec, void **ppUserData);
switch_status_t fork_session_cleanup(switch_core_session_t *session, char *bugname, char* text, int channelIsClosing);
switch_status_t fork_session_stop_play(switch_core_session_t *session, char *bugname);
switch_status_t fork_session_pauseresume(switch_core_session_t *session, char *bugname, int pause);
switch_status_t fork_session_graceful_shutdown(switch_core_session_t *session, char *bugname);
switch_status_t fork_session_send_text(switch_core_session_t *session, char *bugname, char* text);
switch_bool_t fork_frame(switch_core_session_t *session, switch_media_bug_t *bug);
switch_bool_t dub_speech_frame(switch_media_bug_t *bug, private_t * tech_pvt);
switch_status_t fork_service_threads();
switch_status_t fork_session_connect(void **ppUserData);

// Tears down a tech_pvt that was successfully created by fork_session_init() but never
// got as far as being attached to the channel as a media bug (e.g. switch_core_media_bug_add()
// itself failed after fork_session_init() already succeeded). Unlike fork_session_cleanup(),
// this does not require a bug to be registered on the channel - it operates directly on
// *ppUserData. Safe to call any time after fork_session_init() returns success and before
// switch_core_media_bug_add() succeeds; nulls *ppUserData when done.
void fork_session_destroy(void **ppUserData);
#endif
