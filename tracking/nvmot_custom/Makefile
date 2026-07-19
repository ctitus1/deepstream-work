################################################################################
# PLACEHOLDER build for a custom NvMOT low-level tracker library.
#
# The build itself was verified once inside deepstream-work:7.1 (it compiles and
# links, and the resulting .so exports the six NvMOT_* symbols unmangled). What
# it builds is still a skeleton that tracks nothing, and the .so has never been
# loaded by nvtracker. Treat the link flags below as a starting point: they are
# only enough for a source file that does not touch NvBufSurface yet.
#
# Follows the CUDA_VER convention already used in this repo -- see
# scripts/build_yolo_parser.sh, which resolves CUDA 12.6 for DeepStream 7.1 and
# passes it as CUDA_VER to DeepStream-Yolo's Makefile.
#
#   CUDA_VER=12.6 make            build libnvds_custom_tracker.so
#   CUDA_VER=12.6 make clean
#
# The resulting .so is what nvtracker's ll-lib-file property would point at.
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
	$(error "CUDA_VER is not set. DeepStream 7.1 -> CUDA_VER=12.6; see scripts/build_yolo_parser.sh")
endif

CC:= g++

# nvdstracker.h declares a typedef inside struct NvMOTMiscConfig, which is valid
# C++ but not C. This must be compiled as C++.
CFLAGS:= -Wall -std=c++14 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes
CFLAGS+= -I/usr/local/cuda-$(CUDA_VER)/include

LIBS:= -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lstdc++fs
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

# TODO: nvtracker resolves the low-level symbols by dlsym, so no DeepStream
# library needs to be linked for the entry points themselves. Add -lnvbufsurface
# / -lnvbufsurftransform here once the tracker actually touches NvBufSurface
# pixels (i.e. once NvMOT_Query reports numTransforms > 0).

SRCFILES:= $(wildcard *.cpp)
TARGET_LIB:= libnvds_custom_tracker.so
TARGET_OBJS:= $(SRCFILES:.cpp=.o)

all: $(TARGET_LIB)

%.o: %.cpp Makefile
	$(CC) -c $(COMMON) -o $@ $(CFLAGS) $<

$(TARGET_LIB): $(TARGET_OBJS)
	$(CC) -o $@ $(TARGET_OBJS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB) $(TARGET_OBJS)

.PHONY: all clean
