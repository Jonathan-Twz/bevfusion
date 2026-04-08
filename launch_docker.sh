docker run --gpus all -it --shm-size 64g \
  -v /home/wenzhe/wm_ws/nuCarla/nuCarla:/dataset/nuscenes \
  -v /home/wenzhe/wm_ws/WoTE/dataset:/dataset/navsim \
  -v /home/wenzhe/wm_ws/bevfusion:/home/bevfusion \
  -v /media/hdd/wenzhe:/media/hdd/wenzhe \
  --user "$(id -u):$(id -g)" \
  bevfusion:nucarla /bin/bash
