# Standalone Makefile for building mod_vad_silero against an installed FreeSWITCH.
# For in-tree builds inside the FreeSWITCH source tree, use Makefile.am instead.
#
# Usage:
#   make                 # build mod_vad_silero.so
#   make AVX2=1          # build with AVX2 SIMD (default: SSE2 on x86_64, scalar elsewhere)
#   make install         # install into $(FS_MODDIR)
#   make clean
#
# Requires the ONNX Runtime headers/libs installed (onnxruntime_cxx_api.h, libonnxruntime).
# The Silero model is loaded at runtime from /usr/local/share/silero_vad/silero_vad.onnx,
# overridable via the SILERO_VAD_MODEL_PATH environment variable.
#
# Override paths if FreeSWITCH is installed somewhere other than /usr/local/freeswitch:
#   make FS_PREFIX=/opt/freeswitch install

MOD       := mod_vad_silero

FS_PREFIX ?= /usr/local/freeswitch
FS_INC    ?= $(FS_PREFIX)/include/freeswitch
FS_MODDIR ?= $(FS_PREFIX)/mod

CC        ?= gcc
CXX       ?= g++

PKG_CFLAGS := $(shell pkg-config --cflags speexdsp)
PKG_LIBS   := $(shell pkg-config --libs   speexdsp)

# SIMD only applies on x86_64; other arches fall back to the scalar code path.
ARCH      := $(shell uname -m)
SIMD_FLAGS :=
ifneq (,$(filter $(ARCH),x86_64 amd64))
    SIMD_FLAGS := -msse2 -DUSE_SSE2
    ifeq ($(AVX2),1)
        SIMD_FLAGS := -mavx2 -DUSE_AVX2
    endif
endif

COMMON_FLAGS := -fPIC -O2 -Wall -I$(FS_INC) -I. $(PKG_CFLAGS) $(SIMD_FLAGS)

CFLAGS    += $(COMMON_FLAGS)
CXXFLAGS  += $(COMMON_FLAGS) -std=c++17

LDFLAGS   += -shared
LIBS      := $(PKG_LIBS) -lonnxruntime -lstdc++ -lpthread

SRC_C     := mod_vad_silero.c
SRC_CXX   := silero_vad_wrapper.cpp vad_iterator.cpp
OBJS      := $(SRC_C:.c=.o) $(SRC_CXX:.cpp=.o)
TARGET    := $(MOD).so

.PHONY: all install clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

install: $(TARGET)
	install -d $(FS_MODDIR)
	install -m 0755 $(TARGET) $(FS_MODDIR)/
	install -d /usr/local/share/silero_vad

clean:
	rm -f $(OBJS) $(TARGET)
