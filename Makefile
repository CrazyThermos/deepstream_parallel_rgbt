################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

CUDA_VER?=11.4
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

WITH_OPENCV:=1

APP:= deepstream-parallel-app

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=6.3

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

SRCS:= $(wildcard *.c) $(wildcard *.cpp)
SRCS+= $(wildcard /opt/nvidia/deepstream/deepstream/sources/apps/apps-common/src/*.c)
INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0

OBJS:= $(SRCS:.c=.o)
OBJS:= $(OBJS:.cpp=.o)

CFLAGS+= -I../../../includes \
		 -I /opt/nvidia/deepstream/deepstream/sources/apps/apps-common/includes \
		 -I../../../includes/cvcore_headers \
		 -I /usr/include/gstreamer-1.0/ \
		 -I /usr/include/glib-2.0 \
		 -I /usr/include/c++/9 \
		 -I/usr/include/aarch64-linux-gnu/c++/9/ \
		 -I./ \
		 -I /usr/lib/aarch64-linux-gnu/glib-2.0/include \
		 -I /usr/local/cuda-$(CUDA_VER)/include \
		 -I /usr/include/opencv4

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

ifeq ($(WITH_OPENCV),1)
 CFLAGS+= -DWITH_OPENCV
endif
LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lnvdsgst_helper -lgstrtspserver-1.0 -lm \
		-L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvds_yml_parser -lyaml-cpp -lnvds_msgbroker\
		-L/opt/nvidia/deepstream/deepstream/lib/cvcore_libs/ -lnvdsgst_smartrecord\
		-lnvds_batch_jpegenc -lnvbufsurface -lnvbufsurftransform -lnvcv_core -lpthread -lJetsonGPIO\
		-lcuda -ldl -Wl,-rpath,$(LIB_INSTALL_DIR)

ifeq ($(WITH_OPENCV),1)
 LIBS+= -lopencv_imgproc -lopencv_core -lopencv_imgcodecs
endif
all: $(APP)

%.o: %.c $(INCS) Makefile
	$(CC) -c -g -o $@ $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -g -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CXX) -g -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)


