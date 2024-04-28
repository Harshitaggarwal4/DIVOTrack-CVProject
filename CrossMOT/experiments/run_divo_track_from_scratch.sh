#!/bin/bash
#SBATCH -A research
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-01:00:00
#SBATCH --mail-user=bhav.beri@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH -o /home2/maharnav.singhal/DIVOTrack/logs/divotrack_train_from_scratch.out
#SBATCH -c 27
#SBATCH --gres=gpu:3

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CrossMOT

echo "Activated"

cd ~/DIVOTrack/CrossMOT/src

python train.py mot --data_cfg '../src/lib/cfg/divo.json' --resume --gpus 0,1,2 --batch_size 8 --num_epochs 30 --exp_id divotrack_train_from_scratch
# python train.py mot --data_cfg '../src/lib/cfg/divo.json' --load_model "../models/fairmot_dla34.pth" --gpus 0,1,2 --batch_size 8 --num_epochs 30 --exp_id divotrack_train_from_scratch

echo "Done"
