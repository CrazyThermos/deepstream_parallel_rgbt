#!/bin/bash
 
# cd build
# cmake ..
make -j 16
 
# 可视化.dot文件保存路径
export GST_DEBUG_DUMP_DOT_DIR=pic/
export NVDS_PERF_MODE=1
# export NVDS_USE_POSTPROCESS=1
# export NVDS_USE_SGIE=1
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
# 执行
./deepstream-parallel-app ./config/deepstream_rgbt_config.yml 