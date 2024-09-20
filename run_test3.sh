export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
# export DISPLAY=:0
python3 /home/orin/workspace/imu_server/imu_data_send.py &
./deepstream-parallel-app ./config/deepstream_rgbt_config.yml &
# wmctrl -r :ACTIVE: -b add,fullscreen