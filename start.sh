CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes=2 train.py

#tmux new -n run && python YouTube/download_channel.py
#tmux new -n run && python YouTube/segment_audio.py
# Ctrl + B, D