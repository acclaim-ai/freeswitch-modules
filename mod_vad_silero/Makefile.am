include $(top_srcdir)/build/modmake.rulesam
MODNAME=mod_vad_silero

mod_LTLIBRARIES = mod_vad_silero.la
mod_vad_silero_la_SOURCES  = mod_vad_silero.c silero_vad_wrapper.cpp vad_iterator.cpp
mod_vad_silero_la_CFLAGS   = $(AM_CFLAGS)
mod_vad_silero_la_CXXFLAGS = $(AM_CXXFLAGS) -std=c++17

# Add SIMD flags based on autoconf detection
if USE_AVX2
mod_vad_silero_la_CXXFLAGS += -DUSE_AVX2 -mavx2
else
if USE_SSE2
mod_vad_silero_la_CXXFLAGS += -DUSE_SSE2 -msse2
endif
endif

mod_vad_silero_la_LIBADD   = $(switch_builddir)/libfreeswitch.la
mod_vad_silero_la_LDFLAGS  = -avoid-version -module -no-undefined -lonnxruntime
