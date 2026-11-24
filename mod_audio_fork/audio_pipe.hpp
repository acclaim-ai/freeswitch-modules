#ifndef __AUDIO_PIPE_HPP__
#define __AUDIO_PIPE_HPP__

#include <cstdint>
#include <string>
#include <list>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <thread>
#include <vector>

#include <libwebsockets.h>

namespace drachtio {

  class AudioPipe {
  public:
    enum LwsState_t {
      LWS_CLIENT_IDLE,
      LWS_CLIENT_CONNECTING,
      LWS_CLIENT_CONNECTED,
      LWS_CLIENT_FAILED,
      LWS_CLIENT_DISCONNECTING,
      LWS_CLIENT_DISCONNECTED
    };
    enum NotifyEvent_t {
      CONNECT_SUCCESS,
      CONNECT_FAIL,
      CONNECTION_DROPPED,
      CONNECTION_CLOSED_GRACEFULLY,
      MESSAGE,
      BINARY
    };
    typedef void (*log_emit_function)(int level, const char *line);
    typedef void (*notifyHandler_t)(const char *sessionId, const char* bugname, NotifyEvent_t event, const char* message, const char* binary, size_t binary_len );

    struct lws_per_vhost_data {
      struct lws_context *context;
      struct lws_vhost *vhost;
      const struct lws_protocols *protocol;
    };

    static void initialize(const char* protocolName, int loglevel, log_emit_function logger);
    static bool deinitialize();
    static bool lws_service_thread();

    // constructor
    // bufLen: PCM payload capacity in bytes (LWS_PRE and header headroom are added internally)
    AudioPipe(const char* uuid, const char* host, unsigned int port, const char* path, int sslFlags,
      size_t bufLen, size_t minFreespace, const char* username, const char* password, char* bugname,
      int bidirectional_audio, int timestamp_header, notifyHandler_t callback);
    ~AudioPipe();  

    LwsState_t getLwsState(void) { return m_state; }
    void connect(void);
    void bufferForSending(const char* text);
    size_t binarySpaceAvailable(void) {
      return m_audio_buffer_max_len - m_audio_buffer_write_offset;
    }
    size_t binaryMinSpace(void) {
      return m_audio_buffer_min_freespace;
    }
    char * binaryWritePtr(void) { 
      return (char *) m_audio_buffer + m_audio_buffer_write_offset;
    }
    void binaryWritePtrAdd(size_t len) {
      m_audio_buffer_write_offset += len;
    }
    // discard buffered (unsent) audio; leaves room for LWS_PRE and the optional timestamp header
    void binaryWritePtrReset(void) {
      m_audio_buffer_write_offset = dataStartOffset();
    }
    void lockAudioBuffer(void) {
      m_audio_mutex.lock();
    }
    void unlockAudioBuffer(void) ;
    bool hasBasicAuth(void) {
      return !m_username.empty() && !m_password.empty();
    }

    void getBasicAuth(std::string& username, std::string& password) {
      username = m_username;
      password = m_password;
    }

    void do_graceful_shutdown();
    bool isGracefulShutdown(void) {
      return m_gracefulShutdown;
    }

    bool is_bidirectional_audio_stream() {
      return m_bidirectional_audio_stream;
    }

    void close() ;

    // queue an individual binary packet (e.g. OPUS frame) to be sent as a separate WS binary message
    void queueBinaryPacket(const uint8_t* data, size_t len);

    // no default constructor or copying
    AudioPipe() = delete;
    AudioPipe(const AudioPipe&) = delete;
    void operator=(const AudioPipe&) = delete;

  private:
    static std::thread serviceThread;

    static int lws_callback(struct lws *wsi, enum lws_callback_reasons reason, void *user, void *in, size_t len); 
    static struct lws_context *context;
    static std::string protocolName;
    static std::mutex mutex_connects;
    static std::mutex mutex_disconnects;
    static std::mutex mutex_writes;
    static std::list<AudioPipe*> pendingConnects;
    static std::list<AudioPipe*> pendingDisconnects;
    static std::list<AudioPipe*> pendingWrites;
    static log_emit_function logger;

    static std::mutex mapMutex;
    static bool stopFlag;

    static AudioPipe* findAndRemovePendingConnect(struct lws *wsi);
    static AudioPipe* findPendingConnect(struct lws *wsi);
    static void addPendingConnect(AudioPipe* ap);
    static void addPendingDisconnect(AudioPipe* ap);
    static void addPendingWrite(AudioPipe* ap);
    static void processPendingConnects(lws_per_vhost_data *vhd);
    static void processPendingDisconnects(lws_per_vhost_data *vhd);
    static void processPendingWrites(void);
    
    bool connect_client(struct lws_per_vhost_data *vhd);

    // Optional timestamp header prepended to every outgoing binary WS message when
    // enabled (AUDIO_FORK_TIMESTAMP_HEADER channel var). Layout, big-endian:
    //   [0]     version      (TS_HEADER_VERSION)
    //   [1]     header_len   (TS_HEADER_LEN) - readers skip this many bytes
    //   [2..3]  reserved     (0)
    //   [4..7]  seq          uint32, per WS connection, +1 per binary message, starts at 0
    //   [8..15] send_ts_us   uint64, microseconds since Unix epoch, taken right before lws_write
    // The graceful-shutdown zero-length binary frame is never wrapped.
    static const uint8_t TS_HEADER_VERSION = 1;
    static const size_t TS_HEADER_LEN = 16;

    // offset where binary payload starts, in m_audio_buffer and in queued packets:
    // LWS_PRE headroom + optional timestamp header slot
    size_t dataStartOffset(void) const {
      return LWS_PRE + m_ts_header_len;
    }

    // send one binary media message; `msg` points just past LWS_PRE at the header slot
    // (if enabled), `len` covers header slot + payload. Stamps the header, then lws_write.
    int sendBinary(struct lws *wsi, uint8_t *msg, size_t len);

    LwsState_t m_state;
    std::string m_uuid;
    std::string m_host;
    std::string m_bugname;
    unsigned int m_port;
    std::string m_path;
    std::list<std::string> m_metadata_list;
    std::mutex m_text_mutex;
    std::mutex m_audio_mutex;
    int m_sslFlags;
    struct lws *m_wsi;
    uint8_t *m_audio_buffer;
    size_t m_audio_buffer_max_len;
    size_t m_audio_buffer_write_offset;
    size_t m_audio_buffer_min_freespace;
    uint8_t* m_recv_buf;
    uint8_t* m_recv_buf_ptr;
    size_t m_recv_buf_len;
    struct lws_per_vhost_data* m_vhd;
    notifyHandler_t m_callback;
    log_emit_function m_logger;
    std::string m_username;
    std::string m_password;
    bool m_gracefulShutdown;
    bool m_bidirectional_audio_stream;
    std::queue<std::vector<uint8_t>> m_binary_packet_queue;
    const size_t m_ts_header_len;  // 0 (disabled) or TS_HEADER_LEN
    // only read/written on the lws service thread (LWS_CALLBACK_CLIENT_WRITEABLE), so no lock
    uint32_t m_ts_header_seq;
  };

} // namespace drachtio

#endif
