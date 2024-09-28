# GST_DEBUG=3
# export NVDS_ENABLE_LATENCY_MEASUREMENT=1
# sudo systemctl stop gdm
sudo loginctl terminate-seat seat0
sudo modprobe nvidia-drm modeset=1
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export NVDS_USE_DRMSINK=1
export NVDS_RELEASE_MODE=1
# python3 imu_data_send.py &
nsys profile ./deepstream-parallel-app ./config/deepstream_rgbt_config.yml 
# wait
# >> logfile_imu.log &
# >> logfile.log &
# wait