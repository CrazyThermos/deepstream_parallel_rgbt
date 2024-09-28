sudo systemctl stop gdm
sudo loginctl terminate-seat seat0
sudo modprobe nvidia-drm modeset=1