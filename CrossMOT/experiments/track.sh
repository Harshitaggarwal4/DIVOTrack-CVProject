#!/bin/bash
#SBATCH -A research
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-01:00:00
#SBATCH --mail-user=bhav.beri@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH -o /home2/maharnav.singhal/DIVOTrack/logs/tracking.out # To change
#SBATCH -c 9 # To change
#SBATCH --gres=gpu:1 # To change

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CrossMOT

echo "Activated"

cd ~/DIVOTrack/CrossMOT/src

python track.py mot --load_model /scratch/harshit/DivoTrack_Dataset/CrossMOT/model_last/model_last.pth --test_divo --conf_thres 0.5 --reid_dim 512 --single_view_threshold 0.1 --cross_view_threshold 0.3 --exp_name tracking_single01_cross03 --exp_id tracking_single01_cross03

python track.py mot --load_model /scratch/harshit/DivoTrack_Dataset/CrossMOT/model_last/model_last.pth --test_divo --conf_thres 0.5 --reid_dim 512 --single_view_threshold 0.2 --cross_view_threshold 0.3 --exp_name tracking_single02_cross03 --exp_id tracking_single02_cross03

python track.py mot --load_model /scratch/harshit/DivoTrack_Dataset/CrossMOT/model_last/model_last.pth --test_divo --conf_thres 0.5 --reid_dim 512 --single_view_threshold 0.2 --cross_view_threshold 0.5 --exp_name tracking_single02_cross05 --exp_id tracking_single02_cross05
