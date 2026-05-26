# Standalone Makefile for building mod_audio_fork against an installed FreeSWITCH.
# For in-tree builds inside the FreeSWITCH source tree, use Makefile.am instead.
#
# Usage:
#   make                 # build mod_audio_fork.so
#   make AVX2=1          # build with AVX2 SIMD (default: SSE2)
#   make install         # install into $(FS_MODDIR)
#   make clean
#
# Override paths if FreeSWITCH is installed somewhere other than /usr/local/freeswitch:
#   make FS_PREFIX=/opt/freeswitch install

MOD       := mod_audio_fork

FS_PREFIX ?= /usr/local/freeswitch
FS_INC    ?= $(FS_PREFIX)/include/freeswitch
FS_MODDIR ?= $(FS_PREFIX)/mod

CC        ?= gcc
CXX       ?= g++

PKG_CFLAGS := $(shell pkg-config --cflags libwebsockets opus)
PKG_LIBS   := $(shell pkg-config --libs   libwebsockets opus)

SIMD_FLAGS := -msse2 -DUSE_SSE2
ifeq ($(AVX2),1)
    SIMD_FLAGS := -mavx2 -DUSE_AVX2
endif

COMMON_FLAGS := -fPIC -O2 -Wall -I$(FS_INC) $(PKG_CFLAGS) $(SIMD_FLAGS)

CFLAGS    += $(COMMON_FLAGS)
CXXFLAGS  += $(COMMON_FLAGS) -std=c++11

LDFLAGS   += -shared
LIBS      := $(PKG_LIBS) -lspeexdsp -lboost_system -lboost_thread -lstdc++ -lpthread

SRC_C     := mod_audio_fork.c
SRC_CXX   := lws_glue.cpp parser.cpp audio_pipe.cpp vector_math.cpp
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

clean:
	rm -f $(OBJS) $(TARGET)
