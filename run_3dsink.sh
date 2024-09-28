# GST_DEBUG=3
# export NVDS_ENABLE_LATENCY_MEASUREMENT=1
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export NVDS_EVAL_MODE=1
python3 /home/orin/workspace/imu_server/imu_data_send.py &
./deepstream-parallel-app ./config/deepstream_rgbt_config.yml
# wait