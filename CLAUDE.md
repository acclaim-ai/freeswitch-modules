# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is a fork of jambonz's freeswitch-modules, with one directory per FreeSWITCH module. Acclaim actively maintains only `mod_audio_fork`. It forks call audio over a WebSocket to ws2grpc (the sip-proxy repo) and plays back audio and events the server sends. The other `mod_*` directories are upstream code that we don't build.

Consumers pin a **git tag** of this repo (`v1.0.0`, `v1.0.1`, `v1.0.2`, …):
- k8s: the `freeswitch-image` repo, through `ARG FS_MODULES_REF` in its Dockerfile
- VMs: the `sip-deploy` repo, through `freeswitch_modules_version` in the env playbooks. The module is rebuilt only on a build deploy.

A change reaches either one only after you tag a release and bump that pin. Use plain semver tags with no `-rc` suffix.

## Build

There is no test suite, linter or CI here. To check a change, build `mod_audio_fork` against an installed FreeSWITCH (headers in `/usr/local/freeswitch/include/freeswitch`) on Linux:

```
cd mod_audio_fork
make                                    # SSE2; AVX2=1 for AVX2
make FS_PREFIX=/opt/freeswitch install
make clean
```

`Makefile` is the standalone build, and the one freeswitch-image uses. `Makefile.am` is for in-tree builds inside the FreeSWITCH source tree. If you add a source file, add it to both. Dependencies: libwebsockets, libopus, speexdsp, boost_system/boost_thread, and pkg-config entries for `libwebsockets` and `opus`. It won't build on macOS without a FreeSWITCH install, so the real check is the freeswitch-image build.

## mod_audio_fork architecture

There are three layers, and each runs on a different thread:

1. **`mod_audio_fork.c`**: the FreeSWITCH module entry points.
   - Parses the `uuid_audio_fork <uuid> start|stop|send_text|pause|resume|graceful-shutdown|stop_play …` API (`fork_function`).
   - Attaches a media bug per `bugname`.
   - `capture_callback` runs on the **media thread** and calls `fork_frame` on `SWITCH_ABC_TYPE_READ`.
2. **`lws_glue.cpp`**: per-session state (`private_t` in `mod_audio_fork.h`), plus the bridge between FreeSWITCH and the pipe.
   - `fork_session_init`/`fork_data_init` read the channel variables and create the `AudioPipe`.
   - `fork_frame` reads PCM from the bug, then either resamples (speex) into the pipe's binary buffer (PCM mode) or encodes 20 ms 48 kHz OPUS packets and hands them over with `queueBinaryPacket` (OPUS mode). It uses `switch_mutex_trylock`, so it never blocks the media thread.
   - `eventCallback` turns server messages (playAudio, killAudio, transcription, …) into `mod_audio_fork::*` FreeSWITCH events, and decodes incoming bidirectional audio.
3. **`audio_pipe.cpp`/`.hpp`** (`drachtio::AudioPipe`): the libwebsockets client.
   - One or more static **lws service threads** (`MOD_AUDIO_FORK_SERVICE_THREADS`) own every socket.
   - Other threads never write to a socket. They queue work on `pendingConnects`/`pendingDisconnects`/`pendingWrites` and wake the service thread with `lws_cancel_service`.
   - All actual sending happens in `lws_callback` on `LWS_CALLBACK_CLIENT_WRITEABLE`.
   - Members touched only by the service thread are marked as such. Keep it that way, or add locking.

Buffer layout matters: every outgoing binary buffer reserves `LWS_PRE` bytes of headroom, plus the timestamp header when it is enabled (`dataStartOffset()`). Callers write the payload after that offset, and `sendBinary` fills the header in place just before `lws_write`.

### Wire protocol with ws2grpc

The server side lives in the `sip-proxy` repo (`src/ws2grpc_adapter.js`, `src/fs_media_header.js`).

- The first text frame carries the metadata JSON. After that, binary frames carry L16 PCM (it can hold several 20 ms frames per message) or one OPUS packet per message.
- The opt-in channel var `AUDIO_FORK_TIMESTAMP_HEADER=true` puts a 16-byte big-endian header on every outgoing binary message: version, header_len, reserved, seq (u32), send_ts_us (u64). It also adds `X-Audio-Fork-Timestamp-Header: 1` to the WS upgrade request.
- Text frames and the zero-length graceful-shutdown frame are never wrapped.
- Changing this format needs a matching ws2grpc change. Otherwise the header bytes reach the transport layer as audio.
- Bump `TS_HEADER_VERSION` for incompatible changes. Receivers skip `header_len` bytes, so you can append fields without a version bump.

`mod_audio_fork/README.md` documents every API argument, channel variable, environment variable and event. Update it with any interface change.

## Git / PR conventions

- `origin` is `acclaim-ai/freeswitch-modules`, and `upstream` is a third-party fork. Always run `gh pr create --repo acclaim-ai/freeswitch-modules`, or the PR goes to upstream.
- Branches: `sip-<n>-<description>`. Commits: Conventional Commits scoped to the module, e.g. `fix(mod_audio_fork): …`.
- Merge to `main`, then tag a release on the merge commit.
- CODEOWNERS: `@acclaim-ai/reviewers-telephony`.
- `CONTRIBUTING.md` (inherited from upstream) asks for a DCO `Signed-off-by` line (`git commit -s`).
- No AI attribution: no `Co-Authored-By: Claude …` trailer in commits, and no "Generated with Claude Code" line in PR descriptions.

## Licensing

`COPYING.md` grants MIT only when FreeSWITCH is dedicated to jambonz and calls are controlled by the jambonz feature server. We use our own dialplan and ws2grpc, so **AGPL-3.0 applies to us**. In practice:

- Every version we deploy must be pushed to the public `acclaim-ai/freeswitch-modules` repo. Never deploy unpushed module code.
- Keep the modification notice in `mod_audio_fork/README.md` current.
- Talk to legal before giving the FreeSWITCH image or binaries to anyone outside Acclaim (e.g. an on-prem install).
